//use ndarray::{ArrayD, Array2};
#[derive(Clone)]
pub struct Observation<T>
where
    T: std::clone::Clone,
{
    pub s0: T,
    pub a: usize,
    pub s1: T,
    pub r: f32,
}

impl<T: std::clone::Clone> Observation<T> {
    pub fn new(s0: T, a: usize, s1: T, r: f32) -> Self {
        Observation { s0, a, s1, r }
    }
}
