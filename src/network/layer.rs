use crate::network::concrete_layer;
use concrete_layer::dense::DenseLayer;
use concrete_layer::softmax::SoftmaxLayer;
use concrete_layer::sigmoid::SigmoidLayer;
use concrete_layer::flatten::FlattenLayer;
use crate::network::layer_trait::Layer;
use ndarray::ArrayD;

pub enum LayerType {
    F(FlattenLayer),
    D(DenseLayer),
    SO(SoftmaxLayer),
    SI(SigmoidLayer),
}

impl LayerType {
    pub fn new_connection(layer_number: u8, input_dim: usize, output_dim: usize, batch_size: usize, learning_rate: f32) -> Result<Self, String> {
        match layer_number {
            1 => Ok(LayerType::D(DenseLayer::new(input_dim, output_dim, batch_size, learning_rate))),
            _ => Err(format!("Bad Connection Layer: {}", layer_number)),
        }
    }
   
    pub fn new_flatten(input_shape: [usize;3]) -> Result<Self, String> {
      Ok(LayerType::F(FlattenLayer::new(input_shape)))
    }

    pub fn new_activation(layer_number: u8) -> Result<LayerType, String> {
        match layer_number {
            1 => Ok(LayerType::SO(SoftmaxLayer::new())),
            2 => Ok(LayerType::SI(SigmoidLayer::new())),
            _ => Err(format!("Bad Activation Layer: {}", layer_number)),
        }
    }
}

impl Layer for LayerType {
    fn get_type(&self) -> String {
        match self {
            LayerType::D(dense_layer) => dense_layer.get_type(),
            LayerType::SO(softmax_layer) => softmax_layer.get_type(),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.get_type(),
            LayerType::F(flatten_layer) => flatten_layer.get_type(), 
        }
    }
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.forward(input),
            LayerType::SO(softmax_layer) => softmax_layer.forward(input),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.forward(input),
            LayerType::F(flatten_layer) => flatten_layer.forward(input),
        }
    }
    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.backward(feedback),
            LayerType::SO(softmax_layer) => softmax_layer.backward(feedback),
            LayerType::SI(sigmoid_layer) => sigmoid_layer.backward(feedback),
            LayerType::F(flatten_layer) => flatten_layer.backward(feedback),
        }
    }
}
