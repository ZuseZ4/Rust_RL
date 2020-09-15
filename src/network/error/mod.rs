mod bce;
mod cce;
mod noop;
//pub mod mse;

mod error_trait;

pub use bce::BinaryCrossEntropyError;
pub use cce::CategoricalCrossEntropyError;
pub use noop::NoopError;
pub use error_trait::Error;
