use super::Error;
use ndarray::{Array1, ArrayD, Ix1};

/// This function calculates the mean of squares of errors between the neural network output and the ground truth.
#[derive(Clone, Default)]
pub struct MeanSquareError {}

impl MeanSquareError {
    /// No parameters required.
    pub fn new() -> Self {
        MeanSquareError {}
    }
}

impl Error for MeanSquareError {
    fn get_type(&self) -> String {
        format!("Mean Square")
    }

    fn forward(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let output = output.into_dimensionality::<Ix1>().unwrap();
        let target = target.into_dimensionality::<Ix1>().unwrap();
        let n = output.len() as f32;
        let err = output
            .iter()
            .zip(target.iter())
            .fold(0., |err, val| err + f32::powf(val.0 - val.1, 2.))
            / n;
        Array1::<f32>::from_elem(1, 0.5 * err).into_dyn()
    }

    fn backward(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let n = target.len() as f32;
        (output - target) / n
    }

    fn loss_from_logits(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        self.forward(output, target)
    }

    fn deriv_from_logits(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        self.backward(output, target)
    }

    fn clone_box(&self) -> Box<dyn Error> {
        Box::new(self.clone())
    }
}
