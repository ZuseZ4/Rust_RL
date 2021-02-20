use crate::network::optimizer::Optimizer;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::rand_distr::Normal; // not getting Standardnormal to work, which would be nicer
use ndarray_rand::RandomExt;

//#[derive(Clone)]
pub struct ConvCalculator {
    input_shape: (usize, usize),  // (,)
    kernel_shape: (usize, usize), // (num_kernels, elem_per_kernels)
    c: Array3<f32>,
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

impl ConvCalculator {
    pub fn new(
        input_shape: (usize, usize),
        kernel_shape: (usize, usize),
        optimizer: Box<dyn Optimizer>,
        batch_size: usize,
        learning_rate: f32,
    ) -> Result<Self, String> {
        if input_shape.1 != kernel_shape.1 {
            return Err(format!(
                "Those dimensions must match! {} {}",
                input_shape.1, kernel_shape.1
            ));
        }

        let mut w_optimizer = optimizer.clone_box();
        let mut b_optimizer = optimizer;
        w_optimizer.set_input_shape(vec![kernel_shape.0, kernel_shape.1]);
        b_optimizer.set_input_shape(vec![kernel_shape.0]);
        dbg!(eprintln!("kernel shape: {:?}", kernel_shape));

        Ok(ConvCalculator {
            c: Default::default(),
            w: Array::random(kernel_shape, Normal::new(0., 1.).unwrap()),
            b: Array::zeros(kernel_shape.0),
            w_updates: Array::zeros(kernel_shape),
            b_updates: Array::zeros(kernel_shape.0),
            num_in_batch: 0,
            input_shape,
            kernel_shape,
            w_optimizer,
            b_optimizer,
            learning_rate,
            batch_size,
        })
    }

    pub fn clone_box(&self) -> ConvCalculator {
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

    pub fn predict(&self, mut batch_input: Array3<f32>) -> Array3<f32> {
        let batch_size = batch_input.shape()[0];
        let mut output = Array3::zeros((
            batch_size,
            self.kernel_shape.0, // #channel_out == #kernels
            batch_input.shape()[1],
        ));
        //eprintln!("batch_input: {:?}", batch_input.shape());
        for (single_input, mut single_output) in
            batch_input.outer_iter_mut().zip(output.outer_iter_mut())
        {
            //let tmp = single_input.dot(&self.w.t()) + &self.b;
            let tmp = self.w.dot(&single_input.t()).t().into_owned() + &self.b;
            //println!("shape: {:?}", tmp.shape());
            single_output.assign(&tmp.t());
        }
        output
    }

    pub fn forward(&mut self, batch_input: Array3<f32>) -> Array3<f32> {
        self.c = batch_input.clone();
        self.predict(batch_input)
    }

    pub fn backward(&mut self, batch_feedback: Array3<f32>) -> Array3<f32> {
        let batch_size = batch_feedback.shape()[0];
        debug_assert!(
            batch_size == self.c.shape()[0],
            "{} {}",
            batch_size,
            self.c.shape()[0]
        ); // handling other cases becomes quite complex
        let output = Array3::zeros((batch_size, batch_feedback.shape()[1], self.kernel_shape.1));
        for (i, (single_feedback, single_forward)) in batch_feedback
            .outer_iter()
            .zip(self.c.outer_iter())
            .enumerate()
        {
            let w_delta: Array2<f32> = single_feedback.t().dot(&single_forward);
            let b_delta: Array1<f32> = single_feedback.t().dot(&Array1::ones(784)); // bias is always applied
            if self.num_in_batch % self.batch_size == 0 {
                self.w_updates = w_delta;
                self.b_updates = b_delta;
                // next lines is executed on all but the very first visit
                if self.num_in_batch == self.batch_size {
                    update_weights(
                        self.batch_size,
                        self.learning_rate,
                        &self.w_updates,
                        &self.b_updates,
                        &mut self.w,
                        &mut self.b,
                        &mut self.w_optimizer,
                        &mut self.b_optimizer,
                    );
                }
                self.num_in_batch = 1;
            } else {
                self.w_updates += &w_delta;
                self.b_updates += &b_delta;
                self.num_in_batch += 1;
            }
        }
        output // TODO fix feedback for prev_layer
    }
}

fn update_weights(
    batch_size: usize,
    learning_rate: f32,
    w_updates: &Array2<f32>,
    b_updates: &Array1<f32>,
    w: &mut Array2<f32>,
    b: &mut Array1<f32>,
    w_optimizer: &mut Box<dyn Optimizer>,
    b_optimizer: &mut Box<dyn Optimizer>,
) {
    let bs = batch_size as f32;
    *w -= &(w_optimizer.optimize2d(w_updates / bs) * learning_rate);
    *b -= &(b_optimizer.optimize1d(b_updates / bs) * learning_rate);
}
