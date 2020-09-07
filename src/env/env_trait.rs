#[allow(unused_imports)]
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Axis, Ix1};
pub trait Environment {
    fn get_legal_actions(&self) -> Array1<usize>;
    fn step(&self) -> (Array2<f32>, Array1<f32>, f32);
    fn print_board(&self);
    //fn step(&self) -> (String, Vec<usize>, f32);
    //fn get_board_position(&self) -> [i8; 36];
    //fn get_possible_positions(&self) -> (Vec<String>, Vec<usize>);
    //fn get_possible_moves(&self) -> Vec<usize>;
}
