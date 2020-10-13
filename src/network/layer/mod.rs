//! This submodule offers multiple layer implementation.
//!
//! The forward and backward functions have to accept and return data in the form ArrayD\<f32>.
//! Common activation functions are bundled under activation_layer.

mod convolution;
mod dense;
mod dropout;
mod flatten;
//pub mod conv_test;
mod layer_trait;

/// This module contains the most common activation functions like sigmoid, relu, or softmax.
pub mod activation_layer;

/// This trait defines all functions which a layer has to implement to be used as a part of the neural network.
pub use layer_trait::Layer;

/// This layer implements a classical convolution layer.
pub use convolution::ConvolutionLayer;
/// This layer implements a classical dense layer.
pub use dense::DenseLayer;
/// This layer implements a classical dropout layer.
pub use dropout::DropoutLayer;
/// This layer implements a classical dropout layer.
pub use flatten::FlattenLayer;
