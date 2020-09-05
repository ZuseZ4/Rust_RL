use crate::network::layer_trait::Layer;
use ndarray::ArrayD;
pub struct ReLuLayer {}

impl ReLuLayer {
    pub fn new() -> Self {
        ReLuLayer {}
    }
}

impl Layer for ReLuLayer {
    fn get_type(&self) -> String {
        "ReLu Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn predict(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.forward(x)
    }

    fn forward(&mut self, mut x: ArrayD<f32>) -> ArrayD<f32> {
        x.mapv_inplace(|x| if x > 0. { x } else { 0. });
        x
    }

    fn backward(&mut self, mut feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback.mapv_inplace(|x| if x >= 0. { 1. } else { 0. });
        feedback
    }
}
