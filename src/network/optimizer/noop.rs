use super::optimizer_trait::Optimizer;
use ndarray::{Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

#[derive(Default)]
#[derive(Clone)]
pub struct Noop {
}

impl Noop {
  pub fn new() -> Self {
    Noop {
    }
  }

}

impl Optimizer for Noop {
  fn set_input_shape(&mut self, _shape: Vec<usize>) {
  }
  fn optimize(&mut self, delta_w: ArrayD<f32>) -> ArrayD<f32> {
    delta_w
  }
  fn optimize1d(&mut self, delta_w: Array1<f32>) -> Array1<f32> {
    self.optimize(delta_w.into_dyn()).into_dimensionality::<Ix1>().unwrap()
  }
  fn optimize2d(&mut self, delta_w: Array2<f32>) -> Array2<f32> {
    self.optimize(delta_w.into_dyn()).into_dimensionality::<Ix2>().unwrap()
  }
  fn optimize3d(&mut self, delta_w: Array3<f32>) -> Array3<f32> {
    self.optimize(delta_w.into_dyn()).into_dimensionality::<Ix3>().unwrap()
  }
  fn clone(&self) -> Box<dyn Optimizer> {
    Box::new(Clone::clone(self))
  }
}
