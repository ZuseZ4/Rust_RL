use super::Error;
use ndarray::{Array, Array1, ArrayD, azip};

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
        arr.mapv(|x| x.clamp(1e-8, 0.999))
    }
}

impl Error for CategoricalCrossEntropyError {
    fn get_type(&self) -> String {
        format!("Categorical Crossentropy")
    }

    fn loss(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> Array1<f32> {
        debug_assert!(output.shape() == target.shape());
        output.mapv_inplace(|x| x.clamp(1e-8, 0.999));
        let mut loss_arr = Array1::zeros(output.shape()[0]);
        azip!((mut loss in loss_arr.outer_iter_mut(), t in target.outer_iter(), o in output.outer_iter()) {
          let tmp = -(t.into_owned() * o.mapv(f32::ln)).sum();
          loss.fill(tmp);
        });
        loss_arr
    }

    fn deriv(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        debug_assert!(output.shape()[0] == target.shape()[0]);
        -(target / output.mapv(|x| x.clamp(1e-8, 0.999)))
    }

    fn deriv_from_logits(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let max: f32 = *output.clone().max_skipnan();
        output.mapv_inplace(|x| (x - max).exp());
        let sum: f32 = output.iter().filter(|x| !x.is_nan()).sum::<f32>();
        output.mapv_inplace(|x| x / sum);
        output - target
    }

    fn loss_from_logits(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> Array1<f32> {
        debug_assert!(output.shape() == target.shape());
        let mut loss_arr = Array1::zeros(output.shape()[0]);
        azip!((mut loss in loss_arr.outer_iter_mut(), t in target.outer_iter(), o in output.outer_iter()) {
            // ignore nans on sum and max
            let max: f32 = *output.max_skipnan();
            output.mapv_inplace(|x| (x - max).exp());
            let sum: f32 = output.iter().filter(|x| !x.is_nan()).sum::<f32>();
            output.mapv_inplace(|x| x / sum);
            let tmp = -(target * output).iter().sum::<f32>();
            loss.fill(tmp);
        });
        loss_arr
    }

    fn clone_box(&self) -> Box<dyn Error> {
        Box::new(self.clone())
    }
}

//https://gombru.github.io/2018/05/23/cross_entropy_loss/
//https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

//https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
