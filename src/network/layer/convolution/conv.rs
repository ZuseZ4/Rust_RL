use crate::network::optimizer::Optimizer;
use ndarray::{azip, par_azip, Array, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Normal; // not getting Standardnormal to work, which would be nicer
use ndarray_rand::RandomExt;

#[allow(dead_code)]
pub struct ConvCalculator {
    kernel_shape: (usize, usize), // (elements_per_kernels, #kernels)
    c: Array3<f32>,               // copy from the forward for the backward pass
    w: Array2<f32>,
    b: Array1<f32>,
    w_updates: Array2<f32>,
    b_updates: Array1<f32>,
    w_optimizer: Box<dyn Optimizer>,
    b_optimizer: Box<dyn Optimizer>,
    learning_rate: f32,
    num_in_batch: usize,
    batch_size: usize,
}

impl Clone for ConvCalculator {
    fn clone(&self) -> ConvCalculator {
        ConvCalculator {
            w_optimizer: self.w_optimizer.clone_box(),
            b_optimizer: self.b_optimizer.clone_box(),
            c: self.c.clone(),
            w: self.w.clone(),
            b: self.b.clone(),
            w_updates: self.w_updates.clone(),
            b_updates: self.b_updates.clone(),
            ..*self
        }
    }
}

impl ConvCalculator {
    pub fn new(
        kernel_shape: (usize, usize),
        optimizer: Box<dyn Optimizer>,
        batch_size: usize,
        learning_rate: f32,
    ) -> Result<Self, String> {
        let mut w_optimizer = optimizer.clone_box();
        let mut b_optimizer = optimizer;
        w_optimizer.set_input_shape(vec![kernel_shape.0, kernel_shape.1]);
        b_optimizer.set_input_shape(vec![kernel_shape.1]);

        Ok(ConvCalculator {
            c: Default::default(),
            w: Array::random(kernel_shape, Normal::new(0., 1.).unwrap()),
            b: Array::zeros(kernel_shape.1), // one bias value per kernel
            w_updates: Array::zeros(kernel_shape),
            b_updates: Array::zeros(kernel_shape.0),
            num_in_batch: 0,
            kernel_shape,
            w_optimizer,
            b_optimizer,
            learning_rate,
            batch_size,
        })
    }

    // batch_input: (Batch, #applications_per_kernel, elements_per_kernel)
    pub fn predict(&self, batch_input: Array3<f32>) -> Array3<f32> {
        let batch_size = batch_input.shape()[0];
        let mut output = Array3::zeros((
            batch_size,
            self.kernel_shape.1, // #channel_out == #kernels
            batch_input.shape()[1],
        ));
        azip!((single_input in batch_input.outer_iter(), mut single_output in output.outer_iter_mut()) {
            let tmp = single_input.dot(&self.w) + &self.b;
            single_output.assign(&tmp.t());
        });
        output
    }

    // batch_input: (Batch, #applications_per_kernel, elements_per_kernel)
    pub fn forward(&mut self, batch_input: Array3<f32>) -> Array3<f32> {
        self.c = batch_input.clone();
        self.predict(batch_input)
    }

    // feedback: (batch_size, #kernels , applications_per_kernel)
    pub fn backward(&mut self, batch_feedback: Array3<f32>) -> Array3<f32> {
        let batch_size = batch_feedback.shape()[0];
        debug_assert!(batch_size == self.c.shape()[0]);
        //azip!((single_feedback in batch_feedback.outer_iter(), single_forward in self.c.outer_iter()) {
        for (single_feedback, single_forward) in
            batch_feedback.outer_iter().zip(self.c.outer_iter())
        {
            let w_delta: Array2<f32> = single_feedback.dot(&single_forward);
            let b_delta: Array1<f32> = single_feedback.sum_axis(Axis(1)); // bias is always applied
            if self.num_in_batch % self.batch_size == 0 {
                self.w_updates = w_delta;
                self.b_updates = b_delta;
                // next lines is executed on all but the very first visit
                if self.num_in_batch == self.batch_size {
                    let bs = self.batch_size as f32;
                    self.w -= &(self.w_optimizer.optimize2d(&self.w_updates.t() / bs)
                        * self.learning_rate);
                    self.b -=
                        &(self.b_optimizer.optimize1d(&self.b_updates / bs) * self.learning_rate);
                }
                self.num_in_batch = 1;
            } else {
                self.w_updates += &w_delta;
                self.b_updates += &b_delta;
                self.num_in_batch += 1;
            }
        }
        let output = Array3::zeros((batch_size, batch_feedback.shape()[1], self.kernel_shape.1));
        output // TODO fix feedback for prev_layer
    }
}
