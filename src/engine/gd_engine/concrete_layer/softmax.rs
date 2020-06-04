pub struct SoftmaxLayer {
}

impl SoftmaxLayer {
  pub fn new() -> self {
    SoftmaxLayer{}
  }
}

impl Layer for SoftmaxLayer {

  fn get_name(&self) -> String {
    "Softmax Layer".to_string()
  }

  fn softmax_forward(&self, x: Vec<f32>) -> Vec<f32> {
    let max = x.iter().max();
    x.iter().map(|&x| (x-max).exp());
    let sum = x.iter().sum();
    x.iter().map(|&x| x / sum);
    x
  }

  fn softmax_backward(&self, x: Vec<f32>, feedback: Vec<f32>) -> Vec<f32>{
    x.iter().zip(feedback.iter())
      .map(|(&b, &c)| b - c)
      .collect()
  }

}
