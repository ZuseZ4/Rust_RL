use super::Error;
use ndarray::{Array, ArrayD};

use ndarray_stats::QuantileExt;

/// This implements the categorical crossentropy loss.
#[derive(Clone, Default)]
pub struct CategoricalCrossEntropyError {}

impl CategoricalCrossEntropyError {
    /// No parameters required.
    pub fn new() -> Self {
        CategoricalCrossEntropyError {}
    }

    fn clip_values(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        arr.mapv_inplace(|x| if x > 0.9999 { 0.9999 } else { x });
        arr.mapv(|x| if x < 1e-8 { 1e-8 } else { x })
    }
}

impl Error for CategoricalCrossEntropyError {
    fn get_type(&self) -> String {
        format!("Categorical Crossentropy")
    }

    fn forward(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        output = self.clip_values(output);
        let loss = -(target * output.mapv(f32::ln)).sum();
        Array::from_elem(1, loss).into_dyn()
    }

    fn backward(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -(target / self.clip_values(output))
    }

    fn deriv_from_logits(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let max: f32 = *output.clone().max_skipnan();
        output.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = output.iter().filter(|x| !x.is_nan()).sum::<f32>();
        output.mapv_inplace(|x| x / sum);
        output - target
    }

    fn loss_from_logits(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        // ignore nans on sum and max
        let max: f32 = *output.max_skipnan();
        output.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = output.iter().filter(|x| !x.is_nan()).sum::<f32>();
        output.mapv_inplace(|x| x / sum);
        let loss = -(target * output).iter().sum::<f32>();
        Array::from_elem(1, loss).into_dyn()
    }

    fn clone_box(&self) -> Box<dyn Error> {
        Box::new(self.clone())
    }
}

//https://gombru.github.io/2018/05/23/cross_entropy_loss/
//https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

//https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
