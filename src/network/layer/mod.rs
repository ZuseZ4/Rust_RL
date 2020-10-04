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

pub mod activation_layer;

pub use layer_trait::Layer;

pub use convolution::ConvolutionLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use flatten::FlattenLayer;
