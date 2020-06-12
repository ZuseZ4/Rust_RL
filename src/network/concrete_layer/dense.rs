use ndarray::{Array, Array1, Array2, Axis};
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
  pub fn new(input_dim: usize, output_dim: usize, batch_size: usize, l_r: f32) -> Self {
    //xavier init
    //let nn_weights: Array2<f32> = Array::random((input_dim,output_dim), Normal::new(0.0, 1.0).unwrap()) 
    let nn_weights: Array2<f32> = Array::random((output_dim,input_dim), Normal::new(0.0, 1.0).unwrap()) 
      .map(|&x| x / (input_dim as f32).sqrt());
    let nn_bias: Array1<f32> = Array::random(input_dim, Normal::new(0.0, 1.0).unwrap());
    DenseLayer{
      input_dim,
      output_dim,
      learning_rate: l_r,
      weights: nn_weights,
      bias: nn_bias,
      net: Array::zeros((input_dim, batch_size)),
      feedback: Array::zeros((output_dim, batch_size)),
      batch_size,
      predictions: 0,
    }
  }
}



impl Layer for DenseLayer {
  
  fn get_type(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
    let pos_in_batch = self.predictions % self.batch_size;
    //println!("net sum before: {}", self.net.fold(0.0, |sum, val| sum+val));
    self.net.column_mut(pos_in_batch).assign(&x); 
    //println!("net sum after: {}", self.net.fold(0.0, |sum, val| sum+val));
    let last_net = self.net.column(pos_in_batch);
    println!("before");
    let res: Array1<f32> = &self.weights.dot(&last_net) + &self.bias;
    println!("after");
    res
  }

  fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32> {
    
    let pos_in_batch = self.predictions % self.batch_size;
    self.feedback.column_mut(pos_in_batch).assign(&feedback);

    //calc derivate to backprop through layers
    let mut output = Array::zeros(self.input_dim);
    for i in 0..self.input_dim {
      for j in 0..self.output_dim {
        output[i] += feedback[j] * self.weights[[i,j]];
      }
    }



    //store feedback
    self.predictions += 1;
    if self.predictions % self.batch_size == 0 {
      let d_w: Array2<f32> = &self.net.dot(&self.feedback.t()) * self.learning_rate / (self.batch_size as f32);
      let d_b: Array1<f32> = &self.feedback.sum_axis(Axis(1))  * self.learning_rate / (self.batch_size as f32); // verify we sum in the right direction
      
      //println!("weight update sum: {}", d_w.fold(0.0, |sum, val| sum+val));
      //println!("bias update sum: {}", d_b.fold(0.0, |sum, val| sum+val));
      
      self.weights -= &d_w;
      self.bias    -= &d_b; 
    
      println!("weights sum: {}", self.weights.fold(0.0, |sum, val| sum+val));
      println!("bias sum: {}\n", self.bias.fold(0.0, |sum, val| sum+val));
      self.learning_rate *= 0.99;
      self.net = Array::zeros((self.input_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
      self.feedback = Array::zeros((self.output_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
    }
   

    output
    }

}
