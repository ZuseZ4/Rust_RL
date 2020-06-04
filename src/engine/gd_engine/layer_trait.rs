use crate::board::board_trait::BoardInfo;
use ndarray::{Array, Array1};

pub trait Layer {
    fn get_Type(&self) -> String;

    fn forward(&mut self, input: Array1<f32>) -> Array1<f32>;

    fn backward(&mut self, feedback: Array1<f32>) -> Array1<f32>;

    //fn reset_board(&mut self);

    //fn finish_round(&mut self, result: i32);
}
