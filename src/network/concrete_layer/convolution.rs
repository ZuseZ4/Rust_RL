use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be better & faster
use crate::network::layer_trait::Layer;

pub struct ConvolutionLayer {
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

impl ConvolutionLayer {
  pub fn new(filter_shape: (usize, usize), filter_number: usize, batch_size: usize, learning_rate: f32) -> Self { //to start we expect filter.x==filter.y and no bias
    //let nn_weights: Array2<f32> = Array::random((input_dim,output_dim), Normal::new(0.0, 1.0).unwrap()) 
    let weights: Array2<f32> = Array::random((filter_number, filter_shape.0*filter_shape.1), Normal::new(0.0, 1.0 as f32)).unwrap()) 
    ConvolutionLayer{
      input_dim,
      output_dim,
      learning_rate,
      weights,
      net: Array::zeros((input_dim, batch_size)),
      feedback: Array::zeros((output_dim, batch_size)),
      batch_size,
      predictions: 0,
    }
  }

  fn unfold_matrix(x: Array2<f32>, k: usize) {
    let n, m = x.shape();
    let xx: Array2<f32> = Array::zeros(((n - k + 1) * (m - k + 1),k**2));
    let row_num = 0;
    for i in 0..(n-k+1) {
      for j in 0..(m-k+1) {
        xx[row_num] = x[i:i+k, j:j+k];
      }
    }
    xx
  }
}



impl Layer for ConvolutionLayer {
  
  fn get_type(&self) -> String {
    "Convolution Layer".to_string()
  }
  
  fn forward(&mut self, x: Array2<f32>) -> Array3<f32> {
    //let pos_in_batch = self.predictions % self.batch_size; // how to change this?
    //self.net.column_mut(pos_in_batch).assign(&x); 
    let x_unfolded = unfold_matrix(x,self.filter_shape.0);

    //let res: Array3<f32> = Array::zeros(((x.shape().0-filter_shape.0+1,x.shape().1-filter_shape.1+1,self.filter_number)));

    //self.weights.dot(&x) + &self.bias; //was this
    res
  }

  fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32> {
    
    let pos_in_batch = self.predictions % self.batch_size;
    self.feedback.column_mut(pos_in_batch).assign(&feedback);

    //calc derivate to backprop through layers
    let output = self.weights.t().dot(&feedback.t());
    //println!("backprop dense feedback {:?} {:?}",output.shape(), output);
    //println!("weights: {:?}", self.weights);



    //store feedback
    self.predictions += 1;
    if self.predictions % self.batch_size == 0 {
      //let d_w: Array2<f32> = &self.feedback.t().dot(&self.net) * self.learning_rate / (self.batch_size as f32);
      let d_w: Array2<f32> = &self.feedback.dot(&self.net.t()) * self.learning_rate / (self.batch_size as f32);//before
      let d_b: Array1<f32> = &self.feedback.sum_axis(Axis(1))  * self.learning_rate / (self.batch_size as f32); 
      
      //println!("weight update sum: {}", d_w.fold(0.0, |sum, val| sum+val));
      //println!("bias update sum: {}", d_b.fold(0.0, |sum, val| sum+val));
      
      self.weights -= &d_w;
      self.bias    -= &d_b;
    
      //self.learning_rate *= 0.999;
      //println!("new learning rate: {}",self.learning_rate);
      //println!("weights sum: {}", self.weights.fold(0.0, |sum, val| sum+val));
      //println!("bias sum: {}\n", self.bias.fold(0.0, |sum, val| sum+val));
      //println!("weights {}", self.weights);
      //println!("bias {}\n", self.bias);
      self.net = Array::zeros((self.input_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
      self.feedback = Array::zeros((self.output_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
    }
   

    output
    }

}
