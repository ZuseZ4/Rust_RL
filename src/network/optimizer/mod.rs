mod momentum;
mod adagrad;
mod rmsprop;
mod adam;
mod noop;

mod optimizer_trait;
pub use optimizer_trait::Optimizer;

pub use momentum::Momentum;
pub use adagrad::AdaGrad;
pub use rmsprop::RMSProp;
pub use adam::Adam;
pub use noop::Noop;
