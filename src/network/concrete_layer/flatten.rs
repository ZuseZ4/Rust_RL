use crate::network::layer_trait::Layer;
use ndarray::{Array, ArrayD, Ix1};
pub struct FlattenLayer {
  input_shape: [usize;3],
  num_elements: usize,
}

impl FlattenLayer {
  pub fn new(input_shape: [usize;3]) -> Self {
    FlattenLayer{
      input_shape,
      num_elements: input_shape[0]*input_shape[1]*input_shape[2],
    }
  }
}

impl Layer for FlattenLayer {

  fn get_type(&self) -> String {
    "Flatten Layer".to_string()
  }

  fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    x.into_shape(self.num_elements).unwrap().into_dyn()
  }

  fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>{
    feedback.into_shape(self.input_shape).unwrap().into_dyn()
  }

}
