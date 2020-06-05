use crate::network::concrete_layer;
use concrete_layer::dense::DenseLayer;
use concrete_layer::softmax::SoftmaxLayer;
use crate::network::layer_trait::Layer;
use ndarray::Array1;

pub enum LayerType {
    D(DenseLayer),
    S(SoftmaxLayer),
}

impl LayerType {
    pub fn new_connection(layer_number: u8, learning_rate: f32) -> Result<LayerType, String> {
        match layer_number {
            1 => Ok(LayerType::D(DenseLayer::new(learning_rate))),
            _ => Err(format!("Bad Connection Layer: {}", layer_number)),
        }
    }
    
    pub fn new_activation(layer_number: u8) -> Result<LayerType, String> {
        match layer_number {
            1 => Ok(LayerType::S(SoftmaxLayer::new())),
            _ => Err(format!("Bad ACtivation Layer: {}", layer_number)),
        }
    }
}

impl Layer for LayerType {
    fn get_type(&self) -> String {
        match self {
            LayerType::D(dense_layer) => dense_layer.get_type(),
            LayerType::S(softmax_layer) => softmax_layer.get_type(),
        }
    }
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.forward(input),
            LayerType::S(softmax_layer) => softmax_layer.forward(input),
        }
    }
    fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32> {
        match self {
            LayerType::D(dense_layer) => dense_layer.backward(feedback),
            LayerType::S(softmax_layer) => softmax_layer.backward(feedback),
        }
    }
}
