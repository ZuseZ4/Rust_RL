use super::conv_utils;
use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use conv_utils::*;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Axis, Ix3};
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be cleaner & faster
use ndarray_rand::RandomExt;

/// This layer implements a convolution on 2d or 3d input.
pub struct ConvolutionLayer2D {
    batch_size: usize,
    kernels: Array2<f32>,
    in_channels: usize,
    bias: Array1<f32>, // one bias value per kernel
    padding: usize,
    last_input: ArrayD<f32>,
    filter_shape: (usize, usize),
    kernel_updates: Array2<f32>,
    bias_updates: Array1<f32>,
    learning_rate: f32,
    num_in_batch: usize,
    // Rust requires knowledge about obj size during compile time. Optimizers can be set/changed dynamically during runtime, so we just store a reference to the heap
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_shape_feedback1() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ]);
        let output = shape_into_kernel(input);
        assert_eq!(
            output,
            arr2(&[[1., 2., 3., 5., 6., 7.], [9., 10., 11., 13., 14., 15.]])
        );
    }
}

fn new_from_kernels(
    kernels: Array2<f32>,
    bias: Array1<f32>,
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
    filter_shape: (usize, usize),
    in_channels: usize,
    out_channels: usize,
    padding: usize,
    batch_size: usize,
    learning_rate: f32,
) -> ConvolutionLayer2D {
    let elements_per_kernel = filter_shape.0 * filter_shape.1 * in_channels;
    ConvolutionLayer2D {
        filter_shape,
        learning_rate,
        kernels,
        in_channels,
        padding,
        bias,
        last_input: Default::default(),
        kernel_updates: Array::zeros((out_channels, elements_per_kernel)),
        bias_updates: Array::zeros(out_channels),
        batch_size,
        num_in_batch: 0,
        weight_optimizer,
        bias_optimizer,
    }
}

impl ConvolutionLayer2D {
    /// This function prints the kernel values.
    ///
    /// It's main purpose is to analyze the learning success of the first convolution layer.
    /// Later layers might not show clear patterns.
    pub fn print_kernel(&self) {
        let n = self.kernels.nrows();
        println!("printing kernels: \n");
        for i in 0..n {
            let arr = self.kernels.index_axis(Axis(0), i);
            println!(
                "{}\n",
                arr.into_shape((self.in_channels, self.filter_shape.0, self.filter_shape.1))
                    .unwrap()
            );
        }
    }

    /// Allows setting of hand-crafted filters.
    /// 2d or 3d filters have to be reshaped into 1d, so kernels.nrows() equals the amount of kernels used.
    pub fn set_kernels(&mut self, kernels: Array2<f32>) {
        self.kernels = kernels;
    }

    /// Create a new convolution layer.
    ///
    /// Currently we only accept quadratic filter_shapes. Common dimensions are (3,3) or (5,5).
    /// The in_channels has to be set equal to the last dimension of the input images.
    /// The out_channels can be set to any positive value, 16 or 32 might be enough for simple cases or to get started.
    /// The padding will be applied to all sites of the input. Using padding: 1 on a 28x28 image will therefore result in a 30x30 input.
    pub fn new(
        filter_shape: (usize, usize),
        in_channels: usize,
        out_channels: usize,
        padding: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        assert_eq!(
            filter_shape.0, filter_shape.1,
            "currently only supporting quadratic filter!"
        );
        assert!(filter_shape.0 >= 1, "filter_shape has to be one or greater");
        assert!(
            in_channels >= 1,
            "filter depth has to be at least one (and equal to img_channels)"
        );
        let elements_per_kernel = filter_shape.0 * filter_shape.1 * in_channels;
        let kernels: Array2<f32> = Array::random(
            (out_channels, elements_per_kernel),
            Normal::new(0.0, 1.0 as f32).unwrap(),
        );
        assert_eq!(kernels.nrows(), out_channels, "filter implementation wrong");
        let bias = Array::zeros(out_channels); //http://cs231n.github.io/neural-networks-2/
        let mut weight_optimizer = optimizer.clone_box();
        let mut bias_optimizer = optimizer;
        weight_optimizer.set_input_shape(vec![out_channels, elements_per_kernel]);
        bias_optimizer.set_input_shape(vec![out_channels]);
        new_from_kernels(
            kernels,
            bias,
            weight_optimizer,
            bias_optimizer,
            filter_shape,
            in_channels,
            out_channels,
            padding,
            batch_size,
            learning_rate,
        )
    }
}

impl Layer for ConvolutionLayer2D {
    fn get_type(&self) -> String {
        format!("Conv")
    }

    fn get_num_parameter(&self) -> usize {
        self.kernels.nrows() * self.kernels.ncols() + self.kernels.nrows() // num_kernels * size_kernels + bias
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        let out_channels = self.kernels.nrows();
        let new_layer = new_from_kernels(
            self.kernels.clone(),
            self.bias.clone(),
            self.weight_optimizer.clone_box(),
            self.bias_optimizer.clone_box(),
            self.filter_shape,
            self.in_channels,
            out_channels,
            self.padding,
            self.batch_size,
            self.learning_rate,
        );
        Box::new(new_layer)
    }

    fn get_output_shape(&self, input_shape: Vec<usize>) -> Vec<usize> {
        let mut res = vec![self.kernels.nrows(), 0, 0];
        let num_dim = input_shape.len();
        res[1] = input_shape[num_dim - 2] - self.filter_shape.0 + 1 + 2 * self.padding;
        res[2] = input_shape[num_dim - 1] - self.filter_shape.1 + 1 + 2 * self.padding;
        res
    }

    fn predict(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        let tmp = self.get_output_shape(input.shape().to_vec());
        let (output_shape_x, output_shape_y) = (tmp[1], tmp[2]);
        let input = conv_utils::add_padding(self.padding, input);

        // prepare input matrix
        let x_unfolded = unfold_3d_matrix(self.in_channels, input, self.filter_shape.0, true);

        // calculate convolution (=output for next layer)
        let prod = x_unfolded.dot(&self.kernels.t()) + &self.bias;

        // reshape product for next layer: (num_kernels, new_x, new_y)
        let res = fold_output(prod, (self.kernels.nrows(), output_shape_x, output_shape_y));
        res.into_dyn()
    }

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.last_input = conv_utils::add_padding(self.padding, input.clone());
        self.predict(input)
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        // precalculate the weight updates for this batch_element

        // prepare feedback matrix
        // we always return a 3d output in our forward/predict function, so we will always receive a 3d feedback:
        let x: Array3<f32> = feedback.into_dimensionality::<Ix3>().unwrap();
        let feedback_as_kernel = conv_utils::shape_into_kernel(x.clone());

        //prepare feedback
        let k: usize = (feedback_as_kernel.shape()[1] as f64).sqrt() as usize;
        let input_unfolded = unfold_3d_matrix(self.in_channels, self.last_input.clone(), k, false);

        //calculate kernel updates
        let prod = input_unfolded.dot(&feedback_as_kernel.t()).t().into_owned();
        let sum: Array1<f32> = x.sum_axis(Axis(1)).sum_axis(Axis(1)); // 3d feedback, but only 1d bias (1 single f32 bias value per kernel)

        // When having a batch size of 32, we are setting the weight updates once and update them 31 times.
        if self.num_in_batch == 0 {
            self.kernel_updates = prod;
            self.bias_updates = sum;
        } else {
            self.kernel_updates += &prod;
            self.bias_updates += &sum;
        }
        self.num_in_batch += 1;

        // After the final update we finally apply the updates.
        if self.num_in_batch == self.batch_size {
            self.num_in_batch = 0;
            let weight_delta = 1. / (self.batch_size as f32) * self.kernel_updates.clone();
            let bias_delta = 1. / (self.batch_size as f32) * self.bias_updates.clone();
            self.kernels -= &(self.weight_optimizer.optimize2d(weight_delta) * self.learning_rate);
            self.bias -= &(self.bias_optimizer.optimize1d(bias_delta) * self.learning_rate);
            //println!("{:}", self.kernels);
            //println!("{:}", self.kernels.index_axis(Axis(0),3));
            //println!("bias {:}", self.bias);
        }

        // calc feedback for the previous layer:
        // use fliped kernel vectors here
        // https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
        // return that

        Array::zeros(self.last_input.shape()).into_dyn()
    }
}