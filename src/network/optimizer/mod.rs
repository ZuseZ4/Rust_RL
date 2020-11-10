mod adagrad;
mod adam;
mod momentum;
mod noop;
mod rmsprop;

mod optimizer_trait;
pub use optimizer_trait::Optimizer;

pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use momentum::Momentum;
pub use noop::Noop;
pub use rmsprop::RMSProp;
