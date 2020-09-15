mod convolution;
mod dense;
mod dropout;
mod flatten;
//pub mod conv_test;
mod layer_trait;


pub mod activation_layer;

pub use layer_trait::Layer;

pub use convolution::ConvolutionLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use flatten::FlattenLayer;

