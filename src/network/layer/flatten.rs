use crate::network::layer::Layer;
use ndarray::ArrayD;
pub struct FlattenLayer {
    input_shape: Vec<usize>,
    num_elements: usize,
}

impl FlattenLayer {
    pub fn new(input_shape: Vec<usize>) -> Self {
        let num_elements = input_shape.clone().iter().product();
        FlattenLayer {
            input_shape,
            num_elements,
        }
    }
}

impl Layer for FlattenLayer {
    fn get_type(&self) -> String {
        "Flatten Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        vec![input_dim.iter().product()]
    }

    fn predict(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.forward(x)
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        x.into_shape(self.num_elements).unwrap().into_dyn()
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback
            .into_shape(self.input_shape.clone())
            .unwrap()
            .into_dyn()
    }
}
