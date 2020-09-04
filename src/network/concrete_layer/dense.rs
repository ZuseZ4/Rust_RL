use ndarray::{Array, Array1, Array2, ArrayD, Axis, Ix1};
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
  pub fn new(input_dim: usize, output_dim: usize, batch_size: usize, learning_rate: f32) -> Self {
    //xavier init
    let weights: Array2<f32> = Array::random((output_dim,input_dim), Normal::new(0.0, 2.0/((output_dim+input_dim) as f32).sqrt()).unwrap());
    let bias: Array1<f32> = Array::zeros(output_dim); //https://cs231n.github.io/neural-networks-2/#init
    //let bias: Array1<f32> = Array::random((output_dim),Normal::new(0.0, 1.0/(output_dim as f32/2.0)).unwrap());//https://cs231n.github.io/neural-networks-2/#init
    DenseLayer{
      input_dim,
      output_dim,
      learning_rate,
      weights,
      bias,
      net: Array::zeros((input_dim, batch_size)),
      feedback: Array::zeros((output_dim, batch_size)),
      batch_size,
      predictions: 0,
    }
  }
}



impl Layer for DenseLayer {
  
  fn get_type(&self) -> String {
    let output = format!("Dense Layer with {} input and {} output neurons", self.input_dim, self.output_dim);
    output
  }
  
  fn get_output_shape(&self, _input_dim: Vec<usize>) -> Vec<usize> {
    vec![self.output_dim]
  }


  fn predict(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    let input: Array1<f32> = x.into_dimensionality::<Ix1>().unwrap();
    let res: Array1<f32> = self.weights.dot(&input) + &self.bias; 
    res.into_dyn()
  }


  fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    let input: Array1<f32> = x.into_dimensionality::<Ix1>().unwrap();
    let pos_in_batch = self.predictions % self.batch_size;
    self.net.column_mut(pos_in_batch).assign(&input); 
    let res: Array1<f32> = self.weights.dot(&input) + &self.bias; 
    res.into_dyn()
  }


  fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {

    let feedback: Array1<f32> = feedback.into_dimensionality::<Ix1>().unwrap();
    let pos_in_batch = self.predictions % self.batch_size;
    self.feedback.column_mut(pos_in_batch).assign(&feedback);

    //calc derivate to backprop through layers
    let output = self.weights.t().dot(&feedback.t());


    //store feedback
    self.predictions += 1;
    if self.predictions % self.batch_size == 0 {
      let d_w: Array2<f32> = &self.feedback.dot(&self.net.t()) * self.learning_rate / (self.batch_size as f32);
      let d_b: Array1<f32> = &self.feedback.sum_axis(Axis(1))  * self.learning_rate / (self.batch_size as f32); 
      
      assert_eq!( d_w.shape(), self.weights.shape());
      assert_eq!( d_b.shape(), self.bias.shape());

      self.weights -= &d_w;
      self.bias    -= &d_b;
   

      self.net = Array::zeros((self.input_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
      self.feedback = Array::zeros((self.output_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
    }
   

    output.into_dyn()
    }

}
