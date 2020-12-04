use crate::network::layer::Layer;
use ndarray::{Array, ArrayD, Ix1};
pub struct ReshapeLayer {
  input_shape: [usize;3],
  num_elements: usize,
}

impl ReshapeLayer {
  pub fn new(input_shape: [usize;3]) -> Self {
    FlattenLayer{
      input_shape,
      num_elements: input_shape[0]*input_shape[1]*input_shape[2],
    }
  }
}

impl Layer for ReshapeLayer {

  fn get_type(&self) -> String {
    "Reshape".to_string()
  }

  fn get_num_parameter(&self) -> usize {
    0
  }

  fn predict(&self, mut x: ArrayD<f32>) -> ArrayD<f32> {
    self.forward(x)
  }


  fn forward(&self, mut x: ArrayD<f32>) -> ArrayD<f32> {
    x.into_shape(self.num_elements).unwrap().into_dyn()
  }


  fn backward(&self, feedback: ArrayD<f32>) -> ArrayD<f32>{
    feedback.into_shape(self.input_shape).unwrap().into_dyn()
  }

}
