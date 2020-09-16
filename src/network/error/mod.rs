mod bce;
mod cce;
mod noop;
mod mse;
mod rmse;

mod error_trait;
pub use mse::MeanSquareError;
pub use rmse::RootMeanSquareError;
pub use bce::BinaryCrossEntropyError;
pub use cce::CategoricalCrossEntropyError;
pub use noop::NoopError;
pub use error_trait::Error;
