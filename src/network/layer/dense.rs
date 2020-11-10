use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{Array, Array1, Array2, ArrayD, Axis, Ix1};
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
    predictions: usize,
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
        let mut weight_optimizer = optimizer.clone();
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
            predictions: 0,
            weight_optimizer,
            bias_optimizer,
        }
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

    fn predict(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        let input: Array1<f32> = x.into_dimensionality::<Ix1>().unwrap();
        let res: Array1<f32> = self.weights.dot(&input) + &self.bias;
        res.into_dyn()
    }

    fn forward(&mut self, x: ArrayD<f32>) -> ArrayD<f32> {
        let input: Array1<f32> = x.into_dimensionality::<Ix1>().unwrap();
        let pos_in_batch = self.predictions % self.batch_size;
        self.net.column_mut(pos_in_batch).assign(&input);
        let res: Array1<f32> = self.weights.dot(&input) + &self.bias;
        res.into_dyn()
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        let feedback: Array1<f32> = feedback.into_dimensionality::<Ix1>().unwrap();
        let pos_in_batch = self.predictions % self.batch_size;
        self.feedback.column_mut(pos_in_batch).assign(&feedback);

        //calc derivate to backprop through layers
        let output = self.weights.t().dot(&feedback.t());

        //store feedback
        self.predictions += 1;
        if self.predictions % self.batch_size == 0 {
            let d_w: Array2<f32> = &self.feedback.dot(&self.net.t()) / (self.batch_size as f32);
            let d_b: Array1<f32> = &self.feedback.sum_axis(Axis(1)) / (self.batch_size as f32);

            assert_eq!(d_w.shape(), self.weights.shape());
            assert_eq!(d_b.shape(), self.bias.shape());

            self.weights -= &(self.weight_optimizer.optimize2d(d_w) * self.learning_rate);
            self.bias -= &(self.bias_optimizer.optimize1d(d_b) * self.learning_rate);

            self.net = Array::zeros((self.input_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
            self.feedback = Array::zeros((self.output_dim, self.batch_size)); //can be skipped, just ignore/overwrite old vals
        }

        output.into_dyn()
    }
}
