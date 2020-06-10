use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2};
//use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be better & faster
use crate::network::layer_trait::Layer;

pub struct DenseLayer {
  input_dim: usize,
  output_dim: usize,
  learning_rate: f32,
  weights: Array2<f32>,
  bias: Array1<f32>,
  net: Array2<f32>,
  feedback: Array2<f32>,
  batch_size: usize,
  predictions: usize,
}

impl DenseLayer {
  pub fn new(l_r: f32) -> Self {
    //xavier init
    let i_dim: usize = 36;
    let o_dim: usize = 36;
    let bs = 200;
    let nn_weights: Array2<f32> = Array::random((i_dim,o_dim), Normal::new(0.0, 1.0).unwrap()) 
      .map(|&x| x / (i_dim as f32).sqrt());
    let nn_bias: Array1<f32> = Array::random(i_dim, Normal::new(0.0, 1.0).unwrap());
    DenseLayer{
      input_dim: 36,
      output_dim: 36,
      learning_rate: l_r,
      weights: nn_weights,
      bias: nn_bias,
      net: Array::zeros((i_dim, bs)),
      feedback: Array::zeros((o_dim, bs)),
      batch_size: bs,
      predictions: 0,
    }
  }
}



impl Layer for DenseLayer {
  
  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
    let pos = self.predictions%self.batch_size;
    self.net.column_mut(pos)
      .assign(&x); 
    let last_net = self.net.column(pos);
    let res: Array1<f32> = &self.weights.dot(&last_net) + &self.bias;
    res
  }

  fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32> {

    //calc derivate to backprop through layers
    let mut output = Array::zeros(self.input_dim);
    for i in 0..self.input_dim {
      for j in 0..self.output_dim {
        output[i] += feedback[j] * self.weights[[i,j]];
      }
    }

    self.predictions += 1;
    if self.predictions % self.batch_size == 0 {
      let d_w: Array2<f32>  = self.net.dot(&self.feedback.t()) * self.learning_rate / (self.batch_size as f32);
      //let d_b: Array1<f32> = self.feedback.sum() * self.learning_rate / (self.batch_size as f32); // sum is probably the wrong thing
      self.weights = self.weights - d_w;
      //self.bias    = self.bias - d_b; 
    
      println!("weights sum: {}", self.weights.fold(0.0, |sum, val| sum+val));
      self.learning_rate *= 0.99;
      self.net = Array::zeros((self.input_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
      self.feedback = Array::zeros((self.output_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
    }
   

    output
    }

}
