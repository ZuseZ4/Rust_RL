use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{par_azip, Array, Array1, Array2, ArrayD, Axis, Ix2};
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be better & faster
use ndarray_rand::RandomExt;

/// A dense (also called fully connected) layer.
pub struct DenseLayer {
    input_dim: usize,
    output_dim: usize,
    learning_rate: f32,
    weights: Array2<f32>,
    bias: Array1<f32>,
    net: Array2<f32>,
    feedback: Array2<f32>,
    batch_size: usize,
    forward_passes: usize,
    backward_passes: usize,
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
}

impl DenseLayer {
    /// A common constructor for a dense layer.
    ///
    /// The learning_rate is expected to be in the range [0,1].
    /// A batch_size of 1 basically means that no batch processing happens.
    /// A batch_size of 0, a learning_rate outside of [0,1], or an input or output dimension of 0 will result in an error.
    /// TODO: return Result<Self, Error> instead of Self
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        //xavier init
        let weights: Array2<f32> = Array::random(
            (output_dim, input_dim),
            Normal::new(0.0, 2.0 / ((output_dim + input_dim) as f32).sqrt()).unwrap(),
        );
        let bias: Array1<f32> = Array::zeros(output_dim); //https://cs231n.github.io/neural-networks-2/#init
        let mut weight_optimizer = optimizer.clone_box();
        let mut bias_optimizer = optimizer;
        weight_optimizer.set_input_shape(vec![output_dim, input_dim]);
        bias_optimizer.set_input_shape(vec![output_dim]);
        DenseLayer {
            input_dim,
            output_dim,
            learning_rate,
            weights,
            bias,
            net: Array::zeros((input_dim, batch_size)),
            feedback: Array::zeros((output_dim, batch_size)),
            batch_size,
            forward_passes: 0,
            backward_passes: 0,
            weight_optimizer,
            bias_optimizer,
        }
    }

    fn update_weights(&mut self) {
        let d_w: Array2<f32> = &self.feedback.dot(&self.net.t()) / (self.batch_size as f32);
        let d_b: Array1<f32> = &self.feedback.sum_axis(Axis(1)) / (self.batch_size as f32);

        assert_eq!(d_w.shape(), self.weights.shape());
        assert_eq!(d_b.shape(), self.bias.shape());

        self.weights -= &(self.weight_optimizer.optimize2d(d_w) * self.learning_rate);
        self.bias -= &(self.bias_optimizer.optimize1d(d_b) * self.learning_rate);
    }
}

impl Layer for DenseLayer {
    fn get_type(&self) -> String {
        format!("Dense")
    }

    fn get_num_parameter(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim // weights + bias
    }

    fn get_output_shape(&self, _input_dim: Vec<usize>) -> Vec<usize> {
        vec![self.output_dim]
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        let new_layer = DenseLayer {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            weight_optimizer: self.weight_optimizer.clone_box(),
            bias_optimizer: self.bias_optimizer.clone_box(),
            net: self.net.clone(),
            feedback: self.feedback.clone(),
            ..*self
        };
        Box::new(new_layer)
    }

    fn predict(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        debug_assert!(x.ndim() == 2, "expected a 2d input!");
        let batch_input: Array2<f32> = x.into_dimensionality::<Ix2>().unwrap();
        let batch_size = batch_input.shape()[0];
        let mut res = Array2::zeros((batch_size, self.output_dim));
        par_azip!((mut res in res.outer_iter_mut(), input in batch_input.outer_iter()) {
            let single_res = &self.weights.dot(&input) + &self.bias;
            res.assign(&single_res);
        });
        res.into_dyn()
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        debug_assert!(self.feedback.ndim() == 2);
        store_input(
            x.clone(),
            &mut self.net,
            self.batch_size,
            &mut self.forward_passes,
        );
        self.predict(x)
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        debug_assert!(self.feedback.ndim() == 2);

        // store feedback gradients for batch-weightupdates
        store_input(
            feedback.clone(),
            &mut self.feedback,
            self.batch_size,
            &mut self.backward_passes,
        );

        //calc derivate to backprop through layers
        // TODO assure which way the a.dot(b) should be calculated!
        let batch_feedback: Array2<f32> = feedback.into_dimensionality::<Ix2>().unwrap();
        let batch_size = batch_feedback.shape()[0];
        let mut output = Array2::zeros((batch_size, self.input_dim));
        par_azip!((mut res in output.outer_iter_mut(), feedback in batch_feedback.outer_iter()) {
            let single_grad = &self.weights.t().dot(&feedback);
            res.assign(single_grad);
        });

        //update weights
        if self.backward_passes % self.batch_size == 0 {
            self.update_weights();
        }

        output.into_dyn()
    }
}

fn store_input(
    input: ArrayD<f32>,
    buffer: &mut Array2<f32>,
    batch_size: usize,
    start_pos: &mut usize,
) {
    assert!(
        input.shape()[0] + *start_pos <= batch_size,
        format!(
            "Dense Layer: \n accumulated batch size of current and previous input 
            is not aligned to the batch size! 
            batch size: {}, prev batches: {}, current batchsize: {}",
            batch_size, *start_pos, input.shape()[0]
        )
    ); // otherwise still not used buffer entries would be overwritten
    let batch_input = input.into_dimensionality::<Ix2>().unwrap();
    let mut pos_in_buffer = *start_pos;
    for single_input in batch_input.outer_iter() {
        buffer.column_mut(pos_in_buffer).assign(&single_input);
        pos_in_buffer = (pos_in_buffer + 1) % batch_size;
    }
    *start_pos = (*start_pos + batch_input.shape()[0]) % batch_size;
}
