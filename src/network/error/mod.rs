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
