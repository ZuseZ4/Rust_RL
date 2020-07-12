use ndarray::ArrayD;

pub trait Layer {
    fn get_type(&self) -> String;

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32>;
}
