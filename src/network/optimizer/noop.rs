use super::optimizer_trait::Optimizer;
use ndarray::{Array1, Array2, Array3, ArrayD};

/// The basic sgd weight update procedure.
#[derive(Default, Clone)]
pub struct Noop {}

impl Noop {
    /// The basic sgd weight update procedure.
    pub fn new() -> Self {
        Noop {}
    }
}

impl Optimizer for Noop {
    fn get_type(&self) -> String {
        format!("None")
    }
    fn set_input_shape(&mut self, _shape: Vec<usize>) {}
    fn optimize(&mut self, delta_w: ArrayD<f32>) -> ArrayD<f32> {
        delta_w
    }
    fn optimize1d(&mut self, delta_w: Array1<f32>) -> Array1<f32> {
        delta_w
    }
    fn optimize2d(&mut self, delta_w: Array2<f32>) -> Array2<f32> {
        delta_w
    }
    fn optimize3d(&mut self, delta_w: Array3<f32>) -> Array3<f32> {
        delta_w
    }
    fn clone(&self) -> Box<dyn Optimizer> {
        Box::new(Clone::clone(self))
    }
}
