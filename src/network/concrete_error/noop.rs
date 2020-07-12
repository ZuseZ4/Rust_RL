use crate::network::error_trait::Error;
use ndarray::ArrayD;
pub struct NoopError {
}

impl NoopError {
  pub fn new() -> Self {
    NoopError{
    }
  }
}

impl Error for NoopError {

  fn get_type(&self) -> String {
    "Noop Layer".to_string()
  }

  fn forward(&mut self, input: ArrayD<f32>, _target: ArrayD<f32>) -> ArrayD<f32> {
    input
  }

  fn backward(&mut self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>{
    feedback
  }

  fn loss_from_logits(&mut self, input: ArrayD<f32>, _feedback: ArrayD<f32>) -> ArrayD<f32> {
    input
  }

  fn deriv_from_logits(&mut self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
    feedback
  }

}
