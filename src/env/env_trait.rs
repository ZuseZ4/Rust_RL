use ndarray::{Array1, Array2};
pub trait Environment {
    fn done(&self) -> bool;
    fn get_legal_actions(&self) -> Array1<usize>;
    fn step(&self) -> (Array2<f32>, Array1<f32>, f32);
    fn take_action(&mut self, action: usize) -> bool;
    fn render(&self);
    fn reset(&mut self);
    fn eval(&self) -> Vec<i8>;
}
