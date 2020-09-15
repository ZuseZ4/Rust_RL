use ndarray::ArrayD;

pub trait Error {
    fn get_type(&self) -> String;

    fn forward(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    fn backward(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    fn loss_from_logits(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    fn deriv_from_logits(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;
}
