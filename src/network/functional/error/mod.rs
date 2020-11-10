mod bce;
mod cce;
mod mse;
mod noop;
//mod rmse;

mod error_trait;

pub use error_trait::Error;

pub use bce::BinaryCrossEntropyError;

pub use cce::CategoricalCrossEntropyError;

pub use mse::MeanSquareError;

pub use noop::NoopError;

//probably not correct impl
//pub use rmse::RootMeanSquareError;
