use super::Error;
use ndarray::{Array1, ArrayD};

/// This function returns 42 during the forward call and forwards the ground trouth unchanged to the previous layer.
///
/// It is intended for debug purpose only.
#[derive(Clone, Default)]
pub struct NoopError {}

impl NoopError {
    /// No parameters required.
    pub fn new() -> Self {
        NoopError {}
    }
}

impl Error for NoopError {
    fn get_type(&self) -> String {
        "Noop Error function".to_string()
    }

    //printing 42 as obviously useless
    fn forward(&self, _input: ArrayD<f32>, _target: ArrayD<f32>) -> ArrayD<f32> {
        Array1::from_elem(1, 42.).into_dyn()
    }

    fn backward(&self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback
    }

    //printing 42 as obviously useless
    fn loss_from_logits(&self, _input: ArrayD<f32>, _feedback: ArrayD<f32>) -> ArrayD<f32> {
        Array1::from_elem(1, 42.).into_dyn()
    }

    fn deriv_from_logits(&self, _input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        feedback
    }

    fn clone_box(&self) -> Box<dyn Error> {
        Box::new(self.clone())
    }
}
