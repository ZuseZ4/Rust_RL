use ndarray::{Array, ArrayD};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Binomial;
use crate::network::layer_trait::Layer;
//use rand::{Rng, ThreadRng};

pub struct DropoutLayer {
  drop_prob: f64,
  dropout_matrix: ArrayD<f32>,
  //rng: ThreadRng,
}

impl DropoutLayer {
  pub fn new(dropout_prob: f32) -> Self {
    DropoutLayer{
      drop_prob: dropout_prob as f64,
      dropout_matrix: Array::zeros(0).into_dyn(),
      //rng: rand::thread_rng(),
    }
  }
}



impl Layer for DropoutLayer {
  
  fn get_type(&self) -> String {
    let output = format!("Dropout Layer with a {:.2}% drop probability", self.drop_prob*100.);
    output
  }


  fn predict(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    x
  }


  fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
    let weights = Array::random(x.shape(), Binomial::new(1,1.-self.drop_prob).unwrap());
    let weights = weights.mapv(|x| x as f32);
    self.dropout_matrix = weights.clone().into_dyn();
    x*weights
  }


  fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
    feedback*self.dropout_matrix.clone()
  }

}
