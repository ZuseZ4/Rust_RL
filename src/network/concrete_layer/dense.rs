use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
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
    let NN = Array::random((36,36), StandardNormal::new())
      .map(|&x| x / 36.sqrt());
    DenseLayer{
      input_dim: 36,
      output_dim: 36,
      weights: NN,
      forward_out: Array::zeros(36),
    }
  }
}


impl Layer for DenseLayer {
  
  fn get_name(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    self.net = x;
    self.forward_out = self.W1.dot(x);
    self.forward_out
  }

  fn backward(&self, feedback: Array1<f32>) -> Array1<f32> {

    //calc derivate to backprop through layers
    let output = Array::zeros(self.input_dim);
    for i in 0..self.inpud_dim {
      for j in 0..self.output_dim {
        output[i] += feedback[j] * self.weights[[i,j]];
      }
    }

    //update own weights
    for i in 0..self.inpud_dim {
      for j in 0..self.output_dim {
        self.weights[[i,j]] -= self.net[i] * updates[j]; //* learning rate
      }
    }

    //update own bias
    //for j in output.len() {
    //  self.bias[j] -= updates[j] * learning_rate; // net theoretically always 1 on bias
    //}


    output

    }
}
