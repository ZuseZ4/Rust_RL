use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Clone)]
pub struct Momentum {
    previous_delta: ArrayD<f32>,
    decay_rate: f32,
}

impl Momentum {
    /// A basic optimization over sgd. A common value might be 0.9.
    pub fn new(decay_rate: f32) -> Self {
        Momentum {
            previous_delta: Array::zeros(0).into_dyn(),
            decay_rate,
        }
    }
}

impl Optimizer for Momentum {
    fn get_type(&self) -> String {
        format!("Momentum")
    }
    fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.previous_delta = Array::zeros(shape);
    }
    fn optimize(&mut self, mut delta_w: ArrayD<f32>) -> ArrayD<f32> {
        delta_w = delta_w + &self.previous_delta * self.decay_rate;
        self.previous_delta = delta_w.clone();
        delta_w
    }
    fn optimize1d(&mut self, delta_w: Array1<f32>) -> Array1<f32> {
        self.optimize(delta_w.into_dyn())
            .into_dimensionality::<Ix1>()
            .unwrap()
    }
    fn optimize2d(&mut self, delta_w: Array2<f32>) -> Array2<f32> {
        self.optimize(delta_w.into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap()
    }
    fn optimize3d(&mut self, delta_w: Array3<f32>) -> Array3<f32> {
        self.optimize(delta_w.into_dyn())
            .into_dimensionality::<Ix3>()
            .unwrap()
    }
    fn clone(&self) -> Box<dyn Optimizer> {
        Box::new(Clone::clone(self))
    }
}
