use ndarray::ArrayD;

pub trait Layer {
    fn get_type(&self) -> String;

    fn predict(&mut self, input: ArrayD<f32>) -> ArrayD<f32>; // similar to forward(..), but no training is expected on this data. Interim results are therefore not stored

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>;
}
