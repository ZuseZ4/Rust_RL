use crate::network::error_trait::Error;
use ndarray::{ArrayD, Array};

pub struct CategoricalCrossEntropyError {
}

impl CategoricalCrossEntropyError {
  pub fn new() -> Self {
    CategoricalCrossEntropyError{
    }
  }
}



impl Error for CategoricalCrossEntropyError {

  fn get_type(&self) -> String {
    "Categorical Crossentropy Error".to_string()
  }

  fn forward(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    let loss = -(target * output.mapv(|x| f32::ln(x))).sum();
    Array::from_elem(1,loss).into_dyn()
  }

  fn backward(&mut self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32>{
    (1.-&target) / (1.-&output) - target/output
  }

  fn loss_from_logits(&mut self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    // ignore nans on sum and max
    let max: f32 = output.iter().fold(f32::MIN, |acc, &x| if x.is_nan() {acc} else {if acc<=x {x} else {acc}});
    output.mapv_inplace(|x| (x-max).exp());
    let sum: f32 = output.iter().sum();
    let loss = -(target*output).iter().sum::<f32>() + f32::ln(sum);
    Array::from_elem(1,loss).into_dyn()    
  }

  fn deriv_from_logits(&mut self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
    let max: f32 = output.iter().fold(f32::MIN, |acc, &x| if x.is_nan() {acc} else {if acc<=x {x} else {acc}});
    output.mapv_inplace(|x| (x-max).exp());
    let sum: f32 = output.iter().sum();
    output.mapv_inplace(|x| x/sum);
    output - target 
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

