pub struct DenseLayer {
}

impl DenseLayer {
  pub fn new() -> self {
    DenseLayer{}
  }
}


impl Layer for DenseLayer {
  
  fn get_name(&self) -> String {
    "Softmax Layer".to_string()
  }
  
  fn dense_forward(&self, x: Vec<f32>) -> Vec<f32> {
    self.W1.dot(x)
  }

  fn backward(&self, feedback: ) -> {
  }

}
