use crate::network::layer::Layer;
use ndarray::ArrayD;

/// A flatten layer which turns higher dimensional input into a one dimension.
///
/// A one dimensional input remains unchanged.
pub struct FlattenLayer {
    input_ndim: usize,
    input_shape: Vec<usize>,
    batch_input_shape: Vec<usize>,
    num_elements: usize,
}

impl FlattenLayer {
    /// The input_shape is required for the backward pass.
    pub fn new(input_shape: Vec<usize>) -> Self {
        let num_elements = input_shape.clone().iter().product();
        let mut batch_input_shape = vec![0];
        batch_input_shape.extend_from_slice(&input_shape);
        FlattenLayer {
            input_ndim: input_shape.len(),
            input_shape,
            batch_input_shape,
            num_elements,
        }
    }
}

impl Layer for FlattenLayer {
    fn get_type(&self) -> String {
        "Flatten".to_string()
    }

    fn get_num_parameter(&self) -> usize {
        0
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        vec![input_dim.iter().product()]
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(FlattenLayer::new(self.input_shape.clone()))
    }

    fn predict(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        if x.ndim() == self.input_ndim {
            return x.into_shape(self.num_elements).unwrap().into_dyn();
        }
        let batch_size = x.shape()[0];
        x.into_shape((batch_size, self.num_elements))
            .unwrap()
            .into_dyn()
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.predict(x)
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        if feedback.ndim() == 1 {
            return feedback
                .into_shape(self.input_shape.clone())
                .unwrap()
                .into_dyn();
        }
        self.batch_input_shape[0] = feedback.shape()[0];
        feedback
            .into_shape(self.batch_input_shape.clone())
            .unwrap()
            .into_dyn()
    }
}
