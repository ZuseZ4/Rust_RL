use crate::network::error::Error;
use crate::network::layer::activation_layer::SigmoidLayer;
use crate::network::layer::Layer;
use ndarray::{Array, ArrayD};

pub struct BinaryCrossEntropyError {
    activation_function: Box<dyn Layer>,
}

impl Default for BinaryCrossEntropyError {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryCrossEntropyError {
    pub fn new() -> Self {
        BinaryCrossEntropyError {
            activation_function: Box::new(SigmoidLayer::new()),
        }
    }
}

impl Error for BinaryCrossEntropyError {
    fn get_type(&self) -> String {
        format!("Binary Crossentropy")
    }

    // loss after activation function (which probably was sigmoid)
    fn forward(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -target.clone() * input.mapv(f32::ln) - (1. - target) * input.mapv(|x| (1. - x).ln())
    }

    // deriv after activation function (which probably was sigmoid)
    fn backward(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -&target / &input + (1.0 - target) / (1.0 - input)
    }

    // takes input from last dense/conv/.. layer directly, without activation function in between
    //Loss(t,z) = max(z,0) - tz + log(1+ e^(-|z|)), t is label
    fn loss_from_logits(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let tmp = input.mapv(|z| 1. + (-f32::abs(z)).exp());
        let loss = input.mapv(|x| f32::max(0., x)) + tmp.mapv(f32::ln) - input * target.clone();
        let cost: f32 = loss.sum() / target.len() as f32;
        Array::from_elem(1, cost).into_dyn()
    }

    // takes input from last dense/conv/.. layer directly, without activation function in between
    fn deriv_from_logits(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        self.activation_function.forward(input) - target
    }
}

//https://gombru.github.io/2018/05/23/cross_entropy_loss/
//https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

//numeric stable version from here (as in keras/tf):
//https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
