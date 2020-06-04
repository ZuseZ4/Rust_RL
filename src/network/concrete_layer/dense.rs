use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be better & faster
use crate::network::layer_trait::Layer;
pub struct DenseLayer {
  input_dim: usize,
  output_dim: usize,
  weights: Array2<f32>,
  net: Array1<f32>,
  forward_out: Array1<f32>,
}

impl DenseLayer {
  pub fn new() -> Self {
    //xavier init
    let i_dim: usize = 36;
    let o_dim: usize = 36;
    let nn: Array2<f32> = Array::random((i_dim,o_dim), Normal::new(0.0, 1.0).unwrap()) 
      .map(|&x| x / (i_dim as f32).sqrt());
    DenseLayer{
      input_dim: 36,
      output_dim: 36,
      weights: nn,
      net: Array::zeros(i_dim),
      forward_out: Array::zeros(o_dim),
    }
  }
}


impl Layer for DenseLayer {
  
  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
    self.net = x;
    self.forward_out = self.weights.dot(&self.net);
    self.forward_out.clone()
  }

  fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32> {

    //calc derivate to backprop through layers
    let mut output = Array::zeros(self.input_dim);
    for i in 0..self.input_dim {
      for j in 0..self.output_dim {
        output[i] += feedback[j] * self.weights[[i,j]];
      }
    }

    //update own weights
    for i in 0..self.input_dim {
      for j in 0..self.output_dim {
        self.weights[[i,j]] -= self.net[i] * feedback[j]; //* learning rate
      }
    }

    //update own bias
    //for j in output.len() {
    //  self.bias[j] -= feedback[j] * learning_rate; // net theoretically always 1 on bias
    //}

    output
    }

}
