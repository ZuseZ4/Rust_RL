use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Clone)]
pub struct RMSProp {
    previous_sum_squared: ArrayD<f32>,
    decay_rate: f32,
}

impl RMSProp {
    /// A constructor including the decay rate. A common value is 0.9.
    pub fn new(decay_rate: f32) -> Self {
        RMSProp {
            previous_sum_squared: Array::zeros(0).into_dyn(),
            decay_rate,
        }
    }
}

impl Optimizer for RMSProp {
    fn get_type(&self) -> String {
        format!("RMSProp")
    }
    fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.previous_sum_squared = Array::zeros(shape);
    }
    fn optimize(&mut self, delta_w: ArrayD<f32>) -> ArrayD<f32> {
        self.previous_sum_squared = self.previous_sum_squared.clone() * self.decay_rate
            + delta_w.mapv(|x| x.powf(2.)) * (1. - self.decay_rate);
        delta_w / self.previous_sum_squared.mapv(f32::sqrt)
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
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(Clone::clone(self))
    }
}
