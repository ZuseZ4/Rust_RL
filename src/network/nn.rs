use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1};
use crate::network::layer::LayerType;
use crate::network::layer_trait::Layer;

pub struct HyperParameter {
  batch_size: usize,
  learning_rate: f32,
  _gamma: f32,
  _decay_rate: f32,
  _resume: bool,
  _render: bool,
}
impl HyperParameter {
  pub fn new() -> Self {
    HyperParameter{
      batch_size: 10,//00,
      learning_rate: 0.1, //10e-4
      _gamma: 0.99,
      _decay_rate: 0.99,
      _resume: false,
      _render: false,
    }
  }
}


pub struct NeuralNetwork {
  input_dim: usize,
  ll_output_dim: usize,
  h_p: HyperParameter,
  layers: Vec<LayerType>,
  last_input:  ArrayD<f32>,
  last_output: Array1<f32>,
  last_target: Array1<f32>,
}


impl NeuralNetwork {
  pub fn new(input_dim: usize) -> NeuralNetwork {

    NeuralNetwork{
      input_dim,
      ll_output_dim: input_dim,
      layers:  vec![],
      h_p: HyperParameter::new(),
      last_input:  Array::zeros(input_dim).into_dyn(),
      last_output: Array::zeros(input_dim),
      last_target: Array::zeros(input_dim),
    }

  }

  //TODO return type Result<ok(_),err>
  pub fn add_activation(&mut self, layer_kind: &str) {
    match layer_kind {
      "softmax" => self.layers.push(LayerType::new_activation(1).unwrap()),
      "sigmoid" => self.layers.push(LayerType::new_activation(2).unwrap()),
      _ => { println!("unknown activation function. Doing nothing!"); return;},
    }
    // don't change output dim, activation layers don't change dimensions
  }

  //TODO return type Result
  pub fn add_connection(&mut self, layer_kind: &str, output_dim: usize) {
    match layer_kind {
      "dense" => self.layers.push(LayerType::new_connection(1, self.ll_output_dim, output_dim, self.h_p.batch_size, self.h_p.learning_rate).unwrap()),
      _ => println!("unknown type, use \"connection\" or \"activation\". Doing nothing!"),
    }
    self.ll_output_dim = output_dim; // update value to output size of new last layer
  }



}

fn normalize(x: Array1<f32>) -> Array1<f32> {
  x.map(|&x| (x+3.0)/6.0)
}

impl NeuralNetwork {


  pub fn forward1d(&mut self, x: Array1<f32>) -> Array1<f32> {
    self.forward(x.into_dyn())
  }
  pub fn forward2d(&mut self, x: Array2<f32>) -> Array1<f32> {
    self.forward(x.into_dyn())
  }
  pub fn forward3d(&mut self, x: Array3<f32>) -> Array1<f32> {
    self.forward(x.into_dyn())
  }


  pub fn forward(&mut self, x: ArrayD<f32>) -> Array1<f32> {
    self.last_input = x.clone();
    let mut input = x.into_dyn();//normalize(x);
    for i in 0..self.layers.len() {
      input = self.layers[i].forward(input);
    }
    self.last_output = input.into_dimensionality::<Ix1>().unwrap(); //output should be Array1 again
    self.last_output.clone()
  }


  pub fn backward(&mut self, target: Array1<f32>) {
    self.last_target = target.clone();
    let mut fb = (&self.last_output - &target).into_dyn(); //should be correct
    for i in (0..self.layers.len()).rev() {
      fb = self.layers[i].backward(fb);
    }
  }

  pub fn error(&mut self) {
    let mse = self.last_output.iter()
      .zip(self.last_target.iter())
      .fold(0.0, |sum, (&x, &y)| sum + 0.5 * (x-y).powf(2.0));
    println!("MSE: {:.4}, input: {}, expected output: {}, was: {:.3}", mse, self.last_input, self.last_target, self.last_output);
  }


}


//.map(|&x| if x < 0 { 0 } else { x }); //ReLu for multilayer

