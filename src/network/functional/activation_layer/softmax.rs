use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::{Array, ArrayD};
use ndarray_stats::QuantileExt;

/// A softmax layer.
#[derive(Default)]
pub struct SoftmaxLayer {
    output: ArrayD<f32>,
}

impl SoftmaxLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        SoftmaxLayer {
            output: Array::zeros(0).into_dyn(), //will be overwritten
        }
    }
}

impl Functional for SoftmaxLayer {}

impl Layer for SoftmaxLayer {
    fn get_type(&self) -> String {
        "Softmax Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn get_num_parameter(&self) -> usize {
        0
    }

    fn predict(&self, mut x: ArrayD<f32>) -> ArrayD<f32> {
        let max: f32 = *x.max_skipnan();
        x.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = x.iter().filter(|x| !x.is_nan()).sum::<f32>();
        x.mapv_inplace(|x| x / sum);
        x
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        self.output = self.predict(x);
        self.output.clone()
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        &self.output - &feedback
    }
}
