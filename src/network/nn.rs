use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix1};
use crate::network::layer::LayerType;
use crate::network::layer_trait::Layer;
use crate::network::error::ErrorType;
use crate::network::error_trait::Error;

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
      batch_size: 1,
      learning_rate: 0.002, //10e-4
      _gamma: 0.99,
      _decay_rate: 0.99,
      _resume: false,
      _render: false,
    }
  }
  pub fn batch_size(&mut self, batch_size: usize) {
    if batch_size <= 0 {
      eprintln!("batch size should be > 0! Doing nothing!");
      return;
    }
    self.batch_size = batch_size;
  }
  pub fn learning_rate(&mut self, learning_rate: f32) {
    if learning_rate <= 0. {
      eprintln!("learning rate should be > 0! Doing nothing!");
      return;
    }
    self.learning_rate = learning_rate;
  }
}


pub struct NeuralNetwork {
  input_dims: Vec<Vec<usize>>, //each layer takes a  1 to 4-dim input. Store details here
  h_p: HyperParameter,
  layers: Vec<LayerType>,
  error: String, //remove due to error_function
  error_function: ErrorType,
  from_logits: bool,
}


impl NeuralNetwork {

  fn new(error: String) -> Self {
    let error_function;
    match ErrorType::new_error(error.clone()) {
      Ok(error_fun) => error_function = error_fun,
      Err(warning) => {
        eprintln!("{}",warning); 
        error_function = ErrorType::new_noop();
      }
    }
    NeuralNetwork{
      error,
      error_function,
      input_dims: vec![vec![]],
      layers:  vec![],
      h_p: HyperParameter::new(),
      from_logits: false,
    }
  }

  pub fn new1d(input_dim: usize, error: String) -> Self {
    NeuralNetwork{
      input_dims: vec![vec![input_dim]],
      ..NeuralNetwork::new(error)
    }
  }
  pub fn new2d((input_dim1, input_dim2): (usize, usize), error: String) -> Self {
    NeuralNetwork{
      input_dims: vec![vec![input_dim1, input_dim2]],
      ..NeuralNetwork::new(error)
    }
  }
  pub fn new3d((input_dim1,input_dim2,input_dim3): (usize,usize,usize), error: String) -> Self {
    NeuralNetwork{
      input_dims: vec![vec![input_dim1, input_dim2, input_dim3]],
      ..NeuralNetwork::new(error)
    }
  }

  pub fn set_batch_size(&mut self, batch_size: usize) {
    self.h_p.batch_size(batch_size);
  }

  pub fn set_learning_rate(&mut self, learning_rate: f32) {
    self.h_p.learning_rate(learning_rate);
  }

  pub fn add_activation(&mut self, layer_kind: &str) {
    let new_activation = LayerType::new_activation(layer_kind.to_string());
    match new_activation {
      Err(error) => {
        eprintln!("{}",error); 
        return;
      }
      Ok(activation) => {
        self.layers.push(activation);
        self.input_dims.push(self.input_dims.last().unwrap().clone()); // activation layers don't change dimensions
      }
    }
    match (self.error.as_str(), layer_kind) {
      ("bce", "sigmoid") => self.from_logits = true,
      ("cce", "softmax") => self.from_logits = true,
      _ => self.from_logits = false,
    }
  }

  pub fn add_convolution(&mut self, filter_shape: (usize, usize), filter_number: usize) {
    let filter_depth: usize;
    let input_dim = self.input_dims.last().unwrap().clone();
    assert!(input_dim.len() == 2 || input_dim.len() == 3, "only implemented conv for 2d or 3d input!");
    if input_dim.len() == 2 {
      filter_depth = 1;
    } else {
      filter_depth = input_dim[0];
    }
    let conv_layer = LayerType::new_convolution(filter_shape, filter_depth, filter_number, self.h_p.batch_size, self.h_p.learning_rate);
    match conv_layer {
      Err(err) => {
        eprintln!("{}",err);
        return;
      }
      Ok(conv) => {
        self.layers.push(conv);
        let input = self.input_dims.last().unwrap().clone();
        let (mut dim_n, mut dim_m) = (0,1); // expect 2d input 
        if input.len() == 3 { // last two dimensions are relevant, so adjust for 3d input
          dim_n += 1;
          dim_m += 1;
        }
        self.input_dims.push(vec![filter_number, input[dim_n]-filter_shape.0+1, input[dim_m]-filter_shape.1+1]);
      }
    }
    self.from_logits = false;
  }

  pub fn add_dense(&mut self, output_dim: usize) {
    if output_dim <= 0 {
      eprintln!("output dimension should be > 0! Doing nothing!");
      return;
    }
    let input_dims = self.input_dims.last().unwrap();
    if input_dims.len()>1 {
      eprintln!("Dense just accepts 1d input! Doing nothing!");
      return;
    }
    let dense_layer = LayerType::new_connection(input_dims[0], output_dim, self.h_p.batch_size, self.h_p.learning_rate);
    match dense_layer {
      Err(err) => {
        eprintln!("{}",err);
        return;
      }
      Ok(dense) => {
        self.layers.push(dense);
        self.input_dims.push(vec![output_dim]);
      }
    }
    self.from_logits = false;
  }

  pub fn add_dropout(&mut self, dropout_prob: f32) {
    if dropout_prob < 0. || dropout_prob > 1. {
      eprintln!("dropout probability has to be between 0. and 1.");
      return
    }
    let dropout_layer = LayerType::new_dropout(dropout_prob);
    match dropout_layer {
      Err(error) => {
        eprintln!("{}", error);
        return;
      }
      Ok(dropout) => {
        self.layers.push(dropout);
        let output = self.input_dims.last().unwrap().clone();
        self.input_dims.push(output);
      }
    }
    self.from_logits = false;
  }

  pub fn add_flatten(&mut self) {
    let input_dims = self.input_dims.last().unwrap();
    if input_dims.len()==1 {
      eprintln!("Input dimension is already one! Doing nothing!");
      return;
    }
    let flatten_layer = LayerType::new_flatten(input_dims.to_vec());
    match flatten_layer {
      Err(error) => {
        eprintln!("{}",error);
        return;
      }
      Ok(flatten) => {
        self.layers.push(flatten);
        let elements = input_dims.iter().fold(1, |prod, val| prod * val);
        self.input_dims.push(vec![elements]); 
      }
    }
    self.from_logits = false;
  }

}

  

impl NeuralNetwork {

  pub fn print_setup(&self) {
    println!("printing neural network layers");
    for i in 0..self.layers.len() {
      println!("{}",self.layers[i].get_type());
    }
    println!("{}",self.error_function.get_type());
    println!("using from_logits optimization: {}", self.from_logits);
    println!();
  }




  pub fn predict1d(&mut self, input: Array1<f32>) -> Array1<f32> {
    self.predict(input.into_dyn())
  }
  pub fn predict2d(&mut self, input: Array2<f32>) -> Array1<f32> {
    self.predict(input.into_dyn())
  }
  pub fn predict3d(&mut self, input: Array3<f32>) -> Array1<f32> {
    self.predict(input.into_dyn())
  }

  pub fn predict(&mut self, mut input: ArrayD<f32>) -> Array1<f32> {
    for i in 0..self.layers.len() {
      input = self.layers[i].predict(input);
    }
    input.into_dimensionality::<Ix1>().unwrap() //output should be Array1 again
  }

  // fix test to work on something else than just 2D input
  pub fn test(&mut self, input: ArrayD<f32>, target: Array2<f32>) {
    let n = target.len_of(Axis(0));
    let mut loss: Array1<f32> = Array1::zeros(n);
    let mut correct: Array1<f32> = Array1::ones(n);
    let mut i = 0;
    for (current_input, current_fb) in input.outer_iter().zip(target.outer_iter()) {
      let pred = self.predict(current_input.into_owned().into_dyn());
      loss[i] = self.loss_from_prediction(pred.clone(), current_fb.into_owned());

      let best_guess: f32 = (pred.clone() * current_fb).sum();
      
      let num: usize = pred.iter().filter(|&x| *x >= best_guess).count();
      if num != 1 {
        correct[i] = 0.;
      }
      i += 1;
    }
    let avg_loss = loss.iter().sum::<f32>()/(n as f32);
    let acc  = correct.iter().sum::<f32>()/(n as f32);
    println!("avg loss: {}, percentage correct: {}", avg_loss, acc);
  }

  pub fn loss_from_prediction(&mut self, prediction: Array1<f32>, target: Array1<f32>) -> f32 {
    let loss = self.error_function.forward(prediction.into_dyn(), target.into_dyn());
    loss[0] 
  }


  pub fn train1d(&mut self, input: Array1<f32>, target: Array1<f32>) {
    self.train(input.into_dyn(), target)
  }
  pub fn train2d(&mut self, input: Array2<f32>, target: Array1<f32>) {
    self.train(input.into_dyn(), target)
  }
  pub fn train3d(&mut self, input: Array3<f32>, target: Array1<f32>) {
    self.train(input.into_dyn(), target)
  }
  pub fn train(&mut self, input: ArrayD<f32>, target: Array1<f32>) { //maybe return option(accuracy,None) and add a setter to return accuracy?
    let mut input = input.into_dyn();
    let n = self.layers.len();


    // forward pass
    // handle layers 1 till pre-last
    for i in 0..(n-1) { 
      input = self.layers[i].forward(input);
    }



    let mut feedback;
    // handle last layer and error function
    if self.from_logits { 
      //merge last layer with error function
      feedback = self.error_function.deriv_from_logits(input, target.into_dyn());
      // to print error function loss here: println!("{}", self.error_function.loss_from_logits(input, target);
    } else { 
      //evaluate last activation layer and error function seperately
      input = self.layers[n-1].forward(input);
      // to print error function loss here: println!("{}", self.error_function.loss(input, target);
      feedback = self.error_function.backward(input, target.into_dyn());
      feedback = self.layers[n-1].backward(feedback);
    }


    // backward pass
    // handle pre-last till first layer
    for i in (0..(n-1)).rev() {
      feedback = self.layers[i].backward(feedback);
    }

  }


}
