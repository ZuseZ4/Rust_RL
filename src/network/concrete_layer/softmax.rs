use crate::network::layer_trait::Layer;
use ndarray::{Array, Array1, Array2};
pub struct SoftmaxLayer {
}

impl SoftmaxLayer {
  pub fn new() -> Self {
    SoftmaxLayer{}
  }
}

impl Layer for SoftmaxLayer {

  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }

  fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    /*
    let max = x.iter().max();
    x.iter().map(|&x| (x-max).exp());
    let sum = x.iter().sum();
    x.iter().map(|&x| x / sum);
    */
    x
  }

  fn backward(&self, x: Array1<f32>, feedback: Array1<f32>) -> Array1<f32>{
    /*
    x.iter().zip(feedback.iter())
      .map(|(&b, &c)| b - c)
      .collect()
      */
  }

}
