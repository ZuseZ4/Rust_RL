use crate::network::layer::Layer;
use ndarray::{Array, ArrayD};
use ndarray_rand::rand_distr::Binomial;
use ndarray_rand::RandomExt;

/// This layer implements a classical dropout layer.
pub struct DropoutLayer {
    drop_prob: f64,
    dropout_matrix: ArrayD<f32>,
}

impl DropoutLayer {
    /// The dropout probability must be in the range [0,1].
    ///
    /// A dropout probability of 1 results in setting every input value to 0.
    /// A dropout probability of 0 results in forwarding the input without changes.
    /// A dropout probability outside of [0,1] results in an error.
    pub fn new(dropout_prob: f32) -> Self {
        DropoutLayer {
            drop_prob: dropout_prob as f64,
            dropout_matrix: Array::zeros(0).into_dyn(),
        }
    }
}

impl Layer for DropoutLayer {
    fn get_type(&self) -> String {
        let output = format!("Dropout: ~{:.2}%", self.drop_prob * 100.);
        output
    }

    fn get_num_parameter(&self) -> usize {
        0
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn predict(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        x
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        let weights = Array::random(x.shape(), Binomial::new(1, 1. - self.drop_prob).unwrap());
        let weights = weights.mapv(|x| x as f32);
        self.dropout_matrix = weights.clone().into_dyn();
        x * weights
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback * self.dropout_matrix.clone()
    }
}
