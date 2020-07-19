use crate::network::error_trait::Error;
use ndarray::{ArrayD, Array1};
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
    "Noop Error function".to_string()
  }

  //printing 42 as obviously useless
  fn forward(&mut self, _input: ArrayD<f32>, _target: ArrayD<f32>) -> ArrayD<f32> {
    Array1::from_elem(1,42.).into_dyn()
  }

  fn backward(&mut self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>{
    feedback
  }

  //printing 42 as obviously useless
  fn loss_from_logits(&mut self, _input: ArrayD<f32>, _feedback: ArrayD<f32>) -> ArrayD<f32> {
    Array1::from_elem(1,42.).into_dyn()
  }

  fn deriv_from_logits(&mut self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
    feedback
  }

}
