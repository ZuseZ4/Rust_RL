use crate::network::layer_trait::Layer;
use ndarray::{Array, ArrayD};

pub struct SoftmaxLayer {
  output: ArrayD<f32>,
}

impl SoftmaxLayer {
  pub fn new() -> Self {
    SoftmaxLayer{
      output: Array::zeros(0).into_dyn(), //will be overwritten
    }
  }
}



impl Layer for SoftmaxLayer {

  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
    input_dim
  }


  fn predict(&mut self, mut x: ArrayD<f32>) -> ArrayD<f32> {
    // ignore nans on sum and max
    let max: f32 = x.iter().fold(f32::MIN, |acc, &x| if x.is_nan() {acc} else {if acc<=x {x} else {acc}});
    x.mapv_inplace(|x| (x-max).exp());
    let sum: f32 = x.iter().sum();
    x.mapv_inplace(|x| x / sum);
    x
  }

  fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    self.output = self.predict(x);        
    self.output.clone()
  }

  fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>{
    (&self.output - &feedback).clone()
  }

}
