use crate::network::layer_trait::Layer;
//use ndarray::{Array, Array1, Array2};
use ndarray::{Array1, ArrayD};
pub struct SigmoidLayer {
  output: ArrayD<f32>,
}

impl SigmoidLayer {
  pub fn new() -> Self {
    SigmoidLayer{
      output: Array1::zeros(0).into_dyn(),
    }
  }
}

impl Layer for SigmoidLayer {

  fn get_type(&self) -> String {
    "Sigmoid Layer".to_string()
  }

  fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    //println!("{:?}",x.shape());
    self.output = x.map(|&x| 1.0 / (1.0 + (-x).exp()));
    self.output.clone()
  }

  fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>{
    self.output.map(|&x| x * (1.0-x)) * feedback
  }

}
