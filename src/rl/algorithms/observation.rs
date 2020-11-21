use ndarray::{ArrayD, Array2};
#[derive(Clone)]
pub struct Observation {
  pub old_state: ArrayD<f32>,
  pub action: usize,
  pub new_state: ArrayD<f32>,
  pub reward: f32,
}

impl Observation {

  pub fn new(s0: Array2<f32>, a: usize, s1: Array2<f32>, r: f32) -> Self {
    Observation {
      old_state: s0.into_dyn(),
      action: a,
      new_state: s1.into_dyn(),
      reward: r,
    }
  }

}
