use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::ArrayD;

/// A relu layer.
#[derive(Default)]
pub struct ReLuLayer {}

impl ReLuLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        ReLuLayer {}
    }
}

impl Functional for ReLuLayer {}

impl Layer for ReLuLayer {
    fn get_type(&self) -> String {
        "ReLu Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn get_num_parameter(&self) -> usize {
        0
    }

    fn predict(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|x| if x > 0. { x } else { 0. })
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.predict(x)
    }

    fn backward(&mut self, mut feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback.mapv_inplace(|x| if x >= 0. { 1. } else { 0. });
        feedback
    }
}
