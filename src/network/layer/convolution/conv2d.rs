use super::conv_utils;
use super::ConvCalculator;
use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{par_azip, Array2, Array3, Array4, ArrayD, Ix4};

// JUST ACCEPTING 4D INPUT! (Batch, Channels, Width, Height)
// Just use batches of size 1 if you don't like my decision (:

// IO Arrays are expected to have one dim more than mentioned here for the batch.
// (The exact batch size can vary and is therefore not stored here.)
// The gradients have the same shape, so no extra shapes for the backward pass are stored.
#[allow(dead_code)]
/// This layer implements a convolution on 2d or 3d input.
pub struct ConvolutionLayer2D {
    conv_layer: ConvCalculator,
    external_input_shape: (usize, usize, usize), // (channels, width, height), e.g. (3,32,32)
    internal_input_shape: (usize, usize),        // (applications_per_kernel, elements_per_kernel)
    internal_output_shape: (usize, usize),       // (#kernels, applications_per_kernel)
    external_output_shape: (usize, usize, usize), // (#kernels, width, height), e.g. (64,32,32)
    external_kernel_shape: (usize, usize, usize, usize), // (#kernels, channels, width, height)
    internal_kernel_shape: (usize, usize),       // (elements_per_kernel, #kernels )
    padding: Vec<usize>,
}

// From external_output -> internal_output
fn reshape_from_next_layer(x: Array4<f32>, expected_shape: (usize, usize)) -> Array3<f32> {
    let batch_size = x.shape()[0];
    x.into_shape((batch_size, expected_shape.0, expected_shape.1))
        .unwrap()
}

// From internal_output -> external_output
fn reshape_for_next_layer(output: Array3<f32>, dst_shape: (usize, usize, usize)) -> Array4<f32> {
    let input_shape: (usize, usize, usize) =
        (output.shape()[0], output.shape()[1], output.shape()[2]);
    let batch_size = input_shape.0;
    match output.into_shape((batch_size, dst_shape.0, dst_shape.1, dst_shape.2)) {
        Ok(v) => v,
        Err(e) => panic!(
            "Sorry, I messed up something internally. \n {} \n {:?} vs {:?}",
            e, input_shape, dst_shape
        ),
    }
}

// From internal_input -> external_input
fn reshape_for_prev_layer(
    _feedback: Array3<f32>,
    _input_shape: (usize, usize, usize),
) -> Array4<f32> {
    Default::default()
}

// From external_input -> internal_input
fn reshape_from_prev_layer(
    mut input: Array4<f32>,
    kernel_width_height: (usize, usize),
    padding: Vec<usize>,
) -> Array3<f32> {
    input = conv_utils::add_padding(input, padding);
    let input_shape = input.shape();
    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let elements_per_kernel = channels * kernel_width_height.0 * kernel_width_height.1;
    let windows = input.windows([
        1,
        input.shape()[1],
        kernel_width_height.0,
        kernel_width_height.1,
    ]);
    let num_kernel_applications = windows.clone().into_iter().count();
    debug_assert!(num_kernel_applications % batch_size == 0); // should be assured by having first dim of windows = 1
    let shape = (
        batch_size,
        num_kernel_applications / batch_size,
        elements_per_kernel,
    );
    let mut res_2d: Array2<f32> = Array2::zeros((num_kernel_applications, elements_per_kernel));
    for (i, window) in windows.into_iter().enumerate() {
        res_2d
            .row_mut(i)
            .assign(&window.into_owned().into_shape(elements_per_kernel).unwrap());
    }
    let output = match res_2d.into_shape(shape) {
        Ok(v) => v,
        Err(_e) => panic!(format!(
            "Input to conv2d is in illegal shape: {:?} {:?}",
            input.shape(),
            shape
        )),
    };
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn from_next_layer_1() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ]);
        let input = input.into_shape((1, 1, 4, 3)).unwrap();
        let target = arr3(&[[[1., 2., 3., 5., 6., 7.], [9., 10., 11., 13., 14., 15.]]]);
        let output = reshape_from_next_layer(input, (2, 6));
        assert_eq!(output, target);
    }

    #[test]
    fn from_prev_layer_1() {
        let input = arr2(&[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        let input = input.into_shape((1, 1, 4, 4)).unwrap();
        let output = reshape_from_prev_layer(input, (3, 3), vec![0; 4]);
        assert_eq!(
            output,
            arr3(&[[
                [1., 2., 3., 5., 6., 7., 9., 10., 11.],
                [2., 3., 4., 6., 7., 8., 10., 11., 12.],
                [5., 6., 7., 9., 10., 11., 13., 14., 15.],
                [6., 7., 8., 10., 11., 12., 14., 15., 16.]
            ]])
        );
    }
    #[test]
    fn from_prev_layer_2() {
        let input = arr2(&[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        let input = input.into_shape((1, 1, 4, 4)).unwrap();
        let output = reshape_from_prev_layer(input, (2, 2), vec![0; 4]);
        assert_eq!(
            output,
            arr3(&[[
                [1., 2., 5., 6.],
                [2., 3., 6., 7.],
                [3., 4., 7., 8.],
                [5., 6., 9., 10.],
                [6., 7., 10., 11.],
                [7., 8., 11., 12.],
                [9., 10., 13., 14.],
                [10., 11., 14., 15.],
                [11., 12., 15., 16.]
            ]])
        );
    }
    #[test]
    fn from_prev_layer_3() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.], [9., 10., 11.]],
            [[12., 13., 14.], [15., 16., 17.], [18., 19., 20.]],
            [[21., 22., 23.], [24., 25., 26.], [27., 28., 29.]],
        ]);
        let input = input.into_shape((1, 3, 3, 3)).unwrap();
        let output = reshape_from_prev_layer(input, (2, 2), vec![0; 4]);
        assert_eq!(
            output,
            arr3(&[[
                [1., 2., 5., 6., 12., 13., 15., 16., 21., 22., 24., 25.],
                [2., 3., 6., 7., 13., 14., 16., 17., 22., 23., 25., 26.],
                [5., 6., 9., 10., 15., 16., 18., 19., 24., 25., 27., 28.],
                [6., 7., 10., 11., 16., 17., 19., 20., 25., 26., 28., 29.]
            ]])
        );
    }
    #[test]
    fn from_prev_for_prev() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.], [9., 10., 11.]],
            [[12., 13., 14.], [15., 16., 17.], [18., 19., 20.]],
            [[21., 22., 23.], [24., 25., 26.], [27., 28., 29.]],
        ]);
        let input = input.into_shape((1, 3, 4, 4)).unwrap();
        let tmp = reshape_from_prev_layer(input.clone(), (3, 3), vec![0; 4]);
        let output = reshape_for_prev_layer(tmp, (0, 0, 0));
        assert_eq!(input, output);
    }
}

impl ConvolutionLayer2D {
    /// This function prints the kernel values.
    ///
    /// It's main purpose is to analyze the learning success of the first convolution layer.
    /// Later layers might not show clear patterns.
    pub fn get_kernels(&self) -> Array4<f32> {
        Default::default()
    }

    /// Allows setting of hand-crafted filters.
    /// 2d or 3d filters have to be reshaped into 1d, so kernels.nrows() equals the amount of kernels used.
    pub fn set_kernels(&mut self, _kernels: Array4<f32>) -> Result<(), String> {
        //TODO
        Ok(())
    }

    /// Create a new convolution layer.
    ///
    /// The in_channels has to be set equal to the last dimension of the input images.
    /// The out_channels can be set to any positive value, 16 or 32 might be enough for simple cases or to get started.
    /// Currently we use padding=full.
    pub fn new(
        kernels: (usize, usize, usize),
        input_shape: (usize, usize, usize), // we will receive (arbitrary large) batches of this shape => 4d input
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let channels_in = input_shape.0;
        let num_kernels = kernels.0;
        let elements_per_kernel = channels_in * kernels.1 * kernels.2;
        let internal_input_shape = (
            channels_in * input_shape.1 * input_shape.2,
            elements_per_kernel,
        );
        let internal_kernel_shape = (elements_per_kernel, num_kernels);
        let internal_output_shape = (num_kernels, input_shape.1 * input_shape.2);
        let conv_layer =
            ConvCalculator::new(internal_kernel_shape, optimizer, batch_size, learning_rate)
                .unwrap();
        let padding = vec![0, 0, kernels.1 / 2, kernels.2 / 2];
        ConvolutionLayer2D {
            conv_layer,
            padding,
            external_input_shape: input_shape,
            internal_input_shape,
            internal_output_shape,
            external_output_shape: (num_kernels, input_shape.1, input_shape.2),
            external_kernel_shape: (num_kernels, channels_in, kernels.1, kernels.2),
            internal_kernel_shape,
        }
    }
}

impl Layer for ConvolutionLayer2D {
    fn get_type(&self) -> String {
        format!("Conv2D")
    }

    fn get_num_parameter(&self) -> usize {
        self.internal_kernel_shape.0 * (self.internal_kernel_shape.1 + 1) // (includes bias)
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        let layer = ConvolutionLayer2D {
            conv_layer: self.conv_layer.clone(),
            padding: self.padding.clone(),
            ..*self
        };
        Box::new(layer)
    }

    // returns the output shape despite of a (maybe) existing batchsize.
    // So those values are for a single input.
    fn get_output_shape(&self, _input_shape: Vec<usize>) -> Vec<usize> {
        let eos = self.external_output_shape;
        vec![eos.0, eos.1, eos.2] // just change the format
    }

    // TODO predict and forward share to much code, just so that self in predict isn't mut. solve it nicer
    fn predict(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        let input = match input.into_dimensionality::<Ix4>() {
          Ok(v) => v,
          Err(_e) => panic!("Please only enter Arrays shaped (batch, input_channels, width, height). \n Use batch=1 if needed!"),
        };
        let input: Array3<f32> = reshape_from_prev_layer(
            input,
            (self.external_kernel_shape.2, self.external_kernel_shape.3),
            self.padding.clone(),
        );
        // now input: (batch, windows_per_batch_element, elements_per_kernel)
        let result = self.conv_layer.predict(input);
        let output = reshape_for_next_layer(result, self.external_output_shape); // (batch, num_kernels, new_width, new_height)
        output.into_dyn()
    }

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        let input = match input.into_dimensionality::<Ix4>() {
          Ok(v) => v,
          Err(_e) => panic!("Please only enter Arrays shaped (batch, input_channels, width, height). \n Use batch=1 if needed!"),
        };
        let input: Array3<f32> = reshape_from_prev_layer(
            input,
            (self.external_kernel_shape.2, self.external_kernel_shape.3),
            self.padding.clone(),
        );
        // now input: (batch, windows_per_batch_element, elements_per_kernel)
        let result = self.conv_layer.forward(input);
        let output = reshape_for_next_layer(result, self.external_output_shape); // (batch, num_kernels, new_width, new_height)
        output.into_dyn()
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        let feedback = match feedback.into_dimensionality::<Ix4>() {
          Ok(v) => v,
          Err(_e) => panic!("Please only enter Arrays shaped (batch, input_channels, width, height). Use batch=1 if needed!"),
        };
        let feedback: Array3<f32> = reshape_from_next_layer(feedback, self.internal_output_shape); // (batch, windows_per_img, elems_per_kernel)
        debug_assert!((feedback.shape()[1], feedback.shape()[2]) == self.internal_output_shape);
        let result = self.conv_layer.backward(feedback);
        let _output = reshape_for_prev_layer(result, self.external_input_shape);
        // https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
        Default::default()
    }
}
