use crate::network::error_trait::Error;
#[allow(unused_imports)]
use ndarray::{ArrayD, Array, Array1, Ix1};

pub struct MeanSquareError {
}

impl MeanSquareError {
  pub fn new() -> Self {
    MeanSquareError{
    }
  }
}



impl Error for MeanSquareError {

  fn get_type(&self) -> String {
    "Mean Square Error".to_string()
  }

  fn forward(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    let output = output.into_dimensionality::<Ix1>().unwrap();
    let target = target.into_dimensionality::<Ix1>().unwrap();
    let n = output.len() as f32;
    let err = output.iter().zip(target.iter()).fold(0., |err, val| err+f32::powf(val.0-val.1,2.)) / n;
    Array1::<f32>::from_elem(1,err).into_dyn()
  }

  fn backward(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>{

    (1.-&target) / (1.-&output) - target/output
  }

  fn loss_from_logits(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    self.forward(output, target)
  }

  fn deriv_from_logits(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    self.backward(output, target)
  }
}


//https://gombru.github.io/2018/05/23/cross_entropy_loss/
//https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
//pub fn binary_crossentropy(target: Array1<f32>, output: Array1<f32>) -> Array1<f32> { //should be used after sigmoid
  //assert len of output vector = 1
  //-&target / &output; // + (1.0-target) / (1.0-output)
//}



//https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
//pub fn categorical_crossentropy(target: Array1<f32>, output: Array1<f32>) -> Array1<f32> { //should be used after softmax
  //-&target / &output + (1.0-target) / (1.0-output)
  //output - target
//}

