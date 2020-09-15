use crate::network::error::error_trait::Error;
use crate::network::layer::Layer;
use crate::network::layer::activation_layer::SigmoidLayer;
use ndarray::ArrayD;

pub struct BinaryCrossEntropyError {
    activation_function: Box<dyn Layer>,
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
        "Binary Crossentropy Error".to_string()
    }

    // loss after activation function (which probably was sigmoid)
    fn forward(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -target.clone() * input.mapv(|x| x.ln()) - (1. - target) * input.mapv(|x| (1. - x).ln())
    }

    // deriv after activation function (which probably was sigmoid)
    fn backward(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -&target / &input + (1.0 - target) / (1.0 - input)
    }

    // takes input from last dense/conv/.. layer directly, without activation function in between
    //Loss(y,z) = max(z,0) - yz + log(1+ e^(-|z|)), y is label
    fn loss_from_logits(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| f32::max(0., x)) + input.mapv(|x| 1. + (-f32::abs(x)).exp()) - input * target
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
