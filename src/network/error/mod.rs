//! This submodule provides various error function.
//!
//! Beside of the normal forward()/backward() functions some Error functions implement both functions in a *_from_logits() variant.
//! They merge their own implementation with the previous activation function in a numerically more stable way.
//! Better known examples are Softmax+CategoricalCrossEntropy or Sigmoid+BinaryCrossEntropy.
//! When used as part of nn, the appropriate functions are automatically picked.

mod bce;
mod cce;
mod mse;
mod noop;
mod rmse;

mod error_trait;
pub use bce::BinaryCrossEntropyError;
pub use cce::CategoricalCrossEntropyError;
pub use error_trait::Error;
pub use mse::MeanSquareError;
pub use noop::NoopError;
pub use rmse::RootMeanSquareError;
