use ndarray::{Array1, Array2, Array3, ArrayD};

pub trait Optimizer {
    fn set_input_shape(&mut self, shape: Vec<usize>);
    fn get_type(&self) -> String;
    fn optimize(&mut self, weight_update: ArrayD<f32>) -> ArrayD<f32>;
    fn optimize1d(&mut self, weight_update: Array1<f32>) -> Array1<f32>;
    fn optimize2d(&mut self, weight_update: Array2<f32>) -> Array2<f32>;
    fn optimize3d(&mut self, weight_update: Array3<f32>) -> Array3<f32>;
    fn clone(&self) -> Box<dyn Optimizer>;
}
