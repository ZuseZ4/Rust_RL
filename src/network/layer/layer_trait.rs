use ndarray::ArrayD;

pub trait Layer {
    fn get_type(&self) -> String;

    fn get_num_parameter(&self) -> usize;

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize>;

    fn predict(&mut self, input: ArrayD<f32>) -> ArrayD<f32>; // similar to forward(..), but no training is expected on this data. Interim results are therefore not stored

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>;
}
