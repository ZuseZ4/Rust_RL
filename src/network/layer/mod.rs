mod convolution;
mod dense;
mod dropout;
mod flatten;
//pub mod conv_test;
mod layer_trait;

/// This trait defines all functions which a layer has to implement to be used as a part of the neural network.
pub use layer_trait::Layer;

/// This layer implements a classical convolution layer.
pub use convolution::ConvolutionLayer2D;
/// This layer implements a classical dense layer.
pub use dense::DenseLayer;
/// This layer implements a classical dropout layer.
pub use dropout::DropoutLayer;
/// This layer implements a classical dropout layer.
pub use flatten::FlattenLayer;
