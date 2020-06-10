use crate::network::layer_trait::Layer;
use ndarray::{Array, Array1};
pub struct SoftmaxLayer {
  output: Array1<f32>,
}

impl SoftmaxLayer {
  pub fn new() -> Self {
    SoftmaxLayer{
      output: Array::zeros(0), //will be overwritten
    }
  }
}

impl Layer for SoftmaxLayer {

  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }

  fn forward(&mut self, mut x: Array1<f32>) -> Array1<f32> {
    
    let max: f32 = x.iter().fold(0.0, |sum, val| sum+val);
    x = x.iter().map(|&x| (x-max).exp()).collect();
    let sum: f32 = x.iter().fold(0.0, |sum, val| sum+val);
    x = x.iter().map(|&x| x / sum).collect();
    self.output = x;
    
    self.output.clone()
  }

  fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32>{
    
    let output = self.output.iter().zip(feedback.iter())
      .map(|(&b, &c)| b - c)
      .collect();
      
    output
  }

}
