use crate::network::layer::Layer;

/// Layer Interface:  
/// All layers passed to the neural network must implement this trait
///
pub trait Functional: Layer + Send + Sync {}
