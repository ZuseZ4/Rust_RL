use crate::network::concrete_layer;
use concrete_layer::dense::DenseLayer;
use concrete_layer::dropout::DropoutLayer;
use concrete_layer::convolution::ConvolutionLayer;
use concrete_layer::activation_layer::softmax::SoftmaxLayer;
use concrete_layer::activation_layer::sigmoid::SigmoidLayer;
use concrete_layer::flatten::FlattenLayer;
use concrete_layer::activation_layer::relu::ReLuLayer;
use concrete_layer::activation_layer::leakyrelu::LeakyReLuLayer;

use crate::network::layer_trait::Layer;
use ndarray::ArrayD;

pub enum LayerType {
    F(FlattenLayer),
    D(DenseLayer),
    DR(DropoutLayer),
    C(ConvolutionLayer),
    SO(SoftmaxLayer),
    SI(SigmoidLayer),
    R(ReLuLayer),
    L(LeakyReLuLayer),
}

impl LayerType {

    pub fn new_connection(input_dim: usize, output_dim: usize, batch_size: usize, learning_rate: f32) -> Result<Self, String> {
      Ok(LayerType::D(DenseLayer::new(input_dim, output_dim, batch_size, learning_rate)))
    }

    pub fn new_convolution(filter_shape: (usize, usize), filter_depth: usize, filter_number: usize, batch_size: usize, learning_rate: f32) -> Result<Self, String>{
      Ok(LayerType::C(ConvolutionLayer::new(filter_shape, filter_depth, filter_number, batch_size, learning_rate)))
    }
   
    pub fn new_flatten(input_shape: Vec<usize>) -> Result<Self, String> {
      Ok(LayerType::F(FlattenLayer::new(input_shape)))
    }

    pub fn new_dropout(dropout_prob: f32) -> Result<Self, String> {
      Ok(LayerType::DR(DropoutLayer::new(dropout_prob)))
    }

    pub fn new_activation(layer_type: String) -> Result<Self, String>{
        match layer_type.as_str() {
            "softmax" => Ok(LayerType::SO(SoftmaxLayer::new())),
            "sigmoid" => Ok(LayerType::SI(SigmoidLayer::new())),
            "relu"    => Ok(LayerType::R(ReLuLayer::new())),
            "leakyrelu" => Ok(LayerType::L(LeakyReLuLayer::new())),
            _ => Err(format!("Bad Activation Layer: {}", layer_type)),
        }
    }
    
}

impl Layer for LayerType {
    fn get_type(&self) -> String {
        match self {
            LayerType::D(dense_layer) => dense_layer.get_type(),
            LayerType::DR(dropout_layer) => dropout_layer.get_type(),
            LayerType::C(conv_layer) => conv_layer.get_type(),
            LayerType::SO(softmax_layer) => softmax_layer.get_type(),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.get_type(),
            LayerType::F(flatten_layer) => flatten_layer.get_type(), 
            LayerType::L(leaky_relu_layer) => leaky_relu_layer.get_type(),
            LayerType::R(relu_layer) => relu_layer.get_type(),
        }
    }
    fn get_output_shape(&self, input_shape: Vec<usize>) -> Vec<usize> {
        match self {
            LayerType::D(dense_layer) => dense_layer.get_output_shape(input_shape),
            LayerType::DR(dropout_layer) => dropout_layer.get_output_shape(input_shape),
            LayerType::C(conv_layer) => conv_layer.get_output_shape(input_shape),
            LayerType::SO(softmax_layer) => softmax_layer.get_output_shape(input_shape),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.get_output_shape(input_shape),
            LayerType::F(flatten_layer) => flatten_layer.get_output_shape(input_shape), 
            LayerType::L(leaky_relu_layer) => leaky_relu_layer.get_output_shape(input_shape),
            LayerType::R(relu_layer) => relu_layer.get_output_shape(input_shape),
        }
    }
    fn predict(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.predict(input),
            LayerType::DR(dropout_layer) => dropout_layer.predict(input),
            LayerType::C(conv_layer) => conv_layer.predict(input),
            LayerType::SO(softmax_layer) => softmax_layer.predict(input),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.predict(input),
            LayerType::F(flatten_layer) => flatten_layer.predict(input),
            LayerType::L(leaky_relu_layer) => leaky_relu_layer.predict(input),
            LayerType::R(relu_layer) => relu_layer.predict(input),
        }
    }
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.forward(input),
            LayerType::DR(dropout_layer) => dropout_layer.forward(input),
            LayerType::C(conv_layer) => conv_layer.forward(input),
            LayerType::SO(softmax_layer) => softmax_layer.forward(input),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.forward(input),
            LayerType::F(flatten_layer) => flatten_layer.forward(input),
            LayerType::L(leaky_relu_layer) => leaky_relu_layer.forward(input),
            LayerType::R(relu_layer) => relu_layer.forward(input),
        }
    }
    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.backward(feedback),
            LayerType::DR(dropout_layer) => dropout_layer.backward(feedback),
            LayerType::C(conv_layer) => conv_layer.backward(feedback),
            LayerType::SO(softmax_layer) => softmax_layer.backward(feedback),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.backward(feedback),
            LayerType::F(flatten_layer) => flatten_layer.backward(feedback),
            LayerType::L(leaky_relu_layer) => leaky_relu_layer.backward(feedback),
            LayerType::R(relu_layer) => relu_layer.backward(feedback),
        }
    }
}
