use crate::network;
use ndarray::par_azip;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix1, Ix2};
use network::functional::activation_layer::{
    LeakyReLuLayer, ReLuLayer, SigmoidLayer, SoftmaxLayer,
};
use network::functional::error::{
    BinaryCrossEntropyError, CategoricalCrossEntropyError, Error, MeanSquareError, NoopError,
};
//RootMeanSquareError,
use network::layer::{ConvolutionLayer2D, DenseLayer, DropoutLayer, FlattenLayer, Layer};
use network::optimizer::*;

#[derive(Clone)]
enum Mode {
    Eval,
    Train,
}

#[derive(Clone, Default)]
struct HyperParameter {
    batch_size: usize,
    learning_rate: f32,
    _gamma: f32,
    _decay_rate: f32,
    _resume: bool,
    _render: bool,
}

impl HyperParameter {
    pub fn new() -> Self {
        HyperParameter {
            batch_size: 1,
            learning_rate: 0.002, //10e-4
            _gamma: 0.99,
            _decay_rate: 0.99,
            _resume: false,
            _render: false,
        }
    }
    pub fn batch_size(&mut self, batch_size: usize) {
        if batch_size == 0 {
            eprintln!("batch size should be > 0! Doing nothing!");
            return;
        }
        self.batch_size = batch_size;
    }
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        if learning_rate < 0. {
            eprintln!("learning rate should be >= 0! Doing nothing!");
            return;
        }
        self.learning_rate = learning_rate;
    }
    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

// Refactor in NeuralNetwork::constructor and NeuralNetwork::executor?
/// The main neural network class. It stores all relevant information.
///
/// Especially all layers, as well as their input and output shape are stored and verified.
pub struct NeuralNetwork {
    input_dims: Vec<Vec<usize>>, //each layer takes a  1 to 4-dim input. Store details here
    h_p: HyperParameter,
    layers: Vec<Box<dyn Layer>>,
    error: String, //remove due to error_function
    error_function: Box<dyn Error>,
    optimizer_function: Box<dyn Optimizer>,
    from_logits: bool,
    mode: Mode,
}

impl Clone for NeuralNetwork {
    fn clone(&self) -> NeuralNetwork {
        let new_layers: Vec<_> = self.layers.iter().map(|x| x.clone_box()).collect();
        NeuralNetwork {
            input_dims: self.input_dims.clone(),
            h_p: self.h_p.clone(),
            layers: new_layers,
            error: self.error.clone(),
            error_function: self.error_function.clone_box(),
            optimizer_function: self.optimizer_function.clone_box(),
            from_logits: self.from_logits,
            mode: self.mode.clone(),
        }
    }
}

impl NeuralNetwork {
    fn get_activation(activation_type: String) -> Result<Box<dyn Layer>, String> {
        match activation_type.as_str() {
            "softmax" => Ok(Box::new(SoftmaxLayer::new())),
            "sigmoid" => Ok(Box::new(SigmoidLayer::new())),
            "relu" => Ok(Box::new(ReLuLayer::new())),
            "leakyrelu" => Ok(Box::new(LeakyReLuLayer::new())),
            _ => Err(format!("Bad Activation Layer: {}", activation_type)),
        }
    }

    fn get_error(error_type: String) -> Result<Box<dyn Error>, String> {
        match error_type.as_str() {
            "mse" => Ok(Box::new(MeanSquareError::new())),
            //"rmse" => Ok(Box::new(RootMeanSquareError::new())),
            "bce" => Ok(Box::new(BinaryCrossEntropyError::new())),
            "cce" => Ok(Box::new(CategoricalCrossEntropyError::new())),
            "noop" => Ok(Box::new(NoopError::new())),
            _ => Err(format!("Unknown Error Function: {}", error_type)),
        }
    }

    fn get_optimizer(optimizer: String) -> Result<Box<dyn Optimizer>, String> {
        match optimizer.as_str() {
            "adagrad" => Ok(Box::new(AdaGrad::new())),
            "rmsprop" => Ok(Box::new(RMSProp::new(0.9))),
            "momentum" => Ok(Box::new(Momentum::new(0.9))),
            "adam" => Ok(Box::new(Adam::new(0.9, 0.999))),
            "none" => Ok(Box::new(Noop::new())),
            _ => Err(format!("Unknown optimizer: {}", optimizer)),
        }
    }

    fn new(error: String, optimizer: String) -> Self {
        let error_function;
        match NeuralNetwork::get_error(error.clone()) {
            Ok(error_fun) => error_function = error_fun,
            Err(warning) => {
                eprintln!("{}", warning);
                error_function = Box::new(NoopError::new());
            }
        }
        let optimizer_function;
        match NeuralNetwork::get_optimizer(optimizer) {
            Ok(optimizer) => optimizer_function = optimizer,
            Err(warning) => {
                eprintln!("{}", warning);
                optimizer_function = Box::new(Noop::new());
            }
        }

        NeuralNetwork {
            error,
            error_function,
            optimizer_function,
            input_dims: vec![vec![]],
            layers: vec![],
            h_p: HyperParameter::new(),
            from_logits: false,
            mode: Mode::Train,
        }
    }

    /// Sets network to inference mode, dropout and backpropagation/training are disabled.
    pub fn eval_mode(&mut self) {
        self.mode = Mode::Eval;
    }

    /// Sets network to train mode, additional calculations for weight updates might occur.
    pub fn train_mode(&mut self) {
        self.mode = Mode::Train;
    }

    /// A constructor for a neural network which takes 1d input.
    pub fn new1d(input_dim: usize, error: String, optimizer: String) -> Self {
        NeuralNetwork {
            input_dims: vec![vec![input_dim]],
            ..NeuralNetwork::new(error, optimizer)
        }
    }
    /// A constructor for a neural network which takes 2d input.
    pub fn new2d(
        (input_dim1, input_dim2): (usize, usize),
        error: String,
        optimizer: String,
    ) -> Self {
        NeuralNetwork {
            input_dims: vec![vec![input_dim1, input_dim2]],
            ..NeuralNetwork::new(error, optimizer)
        }
    }
    /// A constructor for a neural network which takes 3d input.
    pub fn new3d(
        (input_dim1, input_dim2, input_dim3): (usize, usize, usize),
        error: String,
        optimizer: String,
    ) -> Self {
        NeuralNetwork {
            input_dims: vec![vec![input_dim1, input_dim2, input_dim3]],
            ..NeuralNetwork::new(error, optimizer)
        }
    }

    /// A setter to adjust the optimizer.
    ///
    /// By default, batch sgd is beeing used.
    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
        self.optimizer_function = optimizer;
    }

    /// A setter to adjust the error function.
    ///
    /// Should be picked accordingly to the last layer and the given task.
    pub fn set_error_function(&mut self, error: Box<dyn Error>) {
        self.error_function = error;
    }

    /// A setter to adjust the batch size.
    ///
    /// By default a batch size of 1 is used, which is equal to no batch-processing.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.h_p.batch_size(batch_size);
    }

    /// A setter to adjust the learning rate.
    ///
    /// By default a learning rate of 0.002 is used.
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.h_p.set_learning_rate(learning_rate);
    }

    /// A getter for the learning rate.
    pub fn get_learning_rate(&self) -> f32 {
        self.h_p.get_learning_rate()
    }

    /// This function appends a custom layer to the neural network.
    ///
    /// This function might also be used to add a custom activation function to the neural network.
    pub fn store_layer(&mut self, layer: Box<dyn Layer>) {
        let input_shape = self.input_dims.last().unwrap().clone();
        self.input_dims.push(layer.get_output_shape(input_shape));
        self.layers.push(layer);
    }

    /// This function appends one of the implemented activation function to the neural network.
    pub fn add_activation(&mut self, layer_kind: &str) {
        let new_activation = NeuralNetwork::get_activation(layer_kind.to_string());
        match new_activation {
            Err(error) => {
                eprintln!("{}, doing nothing", error);
                return;
            }
            Ok(layer) => {
                self.store_layer(layer);
            }
        }
        match (self.error.as_str(), layer_kind) {
            ("bce", "sigmoid") => self.from_logits = true,
            ("cce", "softmax") => self.from_logits = true,
            _ => self.from_logits = false,
        }
    }

    /// This function appends a convolution layer to the neural network.
    ///
    /// Currently filter_shape.0 == filter_shape.1 is requred.  
    /// If padding > 0 then the x (and if available y) dimension are padded with zeros at both sides.  
    /// E.g. For padding == 1 and x=y=2 we might receive:   
    /// 0000   
    /// 0ab0   
    /// 0cd0   
    /// 0000   
    /// Currently the backpropagation of the error in the convolution layer is not implemented.   
    /// Adding a convolution layer therefore only works if this layer is the first layer in the neural network.
    pub fn add_convolution(
        &mut self,
        filter_shape: (usize, usize),
        filter_number: usize,
        padding: usize,
    ) {
        let filter_depth: usize;
        let input_dim = self.input_dims.last().unwrap().clone();
        assert!(
            input_dim.len() == 2 || input_dim.len() == 3,
            "only implemented conv for 2d or 3d input! {}",
            input_dim.len()
        );
        if input_dim.len() == 2 {
            filter_depth = 1;
        } else {
            filter_depth = input_dim[0];
        }
        let conv_layer = ConvolutionLayer2D::new(
            filter_shape,
            filter_depth,
            filter_number,
            padding,
            self.h_p.batch_size,
            self.h_p.learning_rate,
            self.optimizer_function.clone_box(),
        );
        self.store_layer(Box::new(conv_layer));
        self.from_logits = false;
    }

    /// This function appends a dense (also called fully_conected) layer to the neural network.
    pub fn add_dense(&mut self, output_dim: usize) {
        if output_dim == 0 {
            eprintln!("output dimension should be > 0! Doing nothing!");
            return;
        }
        let input_dims = self.input_dims.last().unwrap();
        if input_dims.len() > 1 {
            eprintln!("Dense just accepts 1d input! Doing nothing!");
            return;
        }
        let dense_layer = DenseLayer::new(
            input_dims[0],
            output_dim,
            self.h_p.batch_size,
            self.h_p.learning_rate,
            self.optimizer_function.clone_box(),
        );
        self.store_layer(Box::new(dense_layer));
        self.from_logits = false;
    }

    /// This function appends a dropout layer to the neural network.
    ///
    /// The dropout probability has to ly in the range [0,1], where   
    /// 0 means that this layer just outputs zeros,  
    /// 1 means that this layer outputs the (unchanged) input,
    /// any value x in between means that input elements are set to zero with a probability of x.
    pub fn add_dropout(&mut self, dropout_prob: f32) {
        if dropout_prob < 0. || dropout_prob > 1. {
            eprintln!("dropout probability has to be between 0. and 1.");
            return;
        }
        let dropout_layer = DropoutLayer::new(dropout_prob);
        self.store_layer(Box::new(dropout_layer));
        self.from_logits = false;
    }

    /// This function appends a flatten layer to the neural network.
    ///
    /// 1d input remains unchanged.
    /// 2d or higher dimensional input is reshaped into a 1d array.
    pub fn add_flatten(&mut self) {
        let input_dims = self.input_dims.last().unwrap();
        if input_dims.len() == 1 {
            eprintln!("Input dimension is already one! Doing nothing!");
            return;
        }
        let flatten_layer = FlattenLayer::new(input_dims.to_vec());
        self.store_layer(Box::new(flatten_layer));
        self.from_logits = false;
    }

    fn print_separator(&self, separator: &str) {
        println!(
            "{}",
            std::iter::repeat(separator).take(70).collect::<String>()
        );
    }

    /// This function prints an overview of the current neural network.
    pub fn print_setup(&self) {
        println!(
            "\nModel: \"sequential\" {:>20} Input shape: {:?}",
            "", self.input_dims[0]
        );
        self.print_separator("─");
        println!(
            "{:<20} {:^20} {:>20}",
            "Layer (type)".to_string(),
            "Output Shape".to_string(),
            "Param #".to_string()
        );
        self.print_separator("═");
        for i in 0..self.layers.len() {
            let layer_type = format!("{:<20}", self.layers[i].get_type());
            let layer_params = format!("{:>20}", self.layers[i].get_num_parameter());
            let output_shape = format!("{:?}", &self.input_dims[i + 1]);
            println!("{} {:^20} {}", layer_type, output_shape, layer_params);
            self.print_separator("─");
        }
        println!(
            "{:<35} {:>30}",
            "Error function:".to_string(),
            self.error_function.get_type()
        );
        println!(
            "{:<35} {:>30}",
            "Optimizer:".to_string(),
            self.optimizer_function.get_type()
        );
        println!(
            "{:<35} {:>30}",
            "using from_logits optimization:".to_string(),
            self.from_logits
        );
        self.print_separator("─");
    }

    /// This function handles the inference on 1d input.
    pub fn predict1d(&self, input: Array1<f32>) -> Array1<f32> {
        self.predict(input.into_dyn())
    }
    /// This function handles the inference on 2d input.
    pub fn predict2d(&self, input: Array2<f32>) -> Array1<f32> {
        self.predict(input.into_dyn())
    }
    /// This function handles the inference on 3d input.
    pub fn predict3d(&self, input: Array3<f32>) -> Array1<f32> {
        self.predict(input.into_dyn())
    }

    /// This function handles the inference on dynamic-dimensional input.
    pub fn predict(&self, mut input: ArrayD<f32>) -> Array1<f32> {
        for i in 0..self.layers.len() {
            input = self.layers[i].predict(input);
        }
        input.into_dimensionality::<Ix1>().unwrap() //output should be Array1 again
    }

    /// This function handles the inference on a batch of dynamic-dimensional input.
    pub fn predict_batch(&self, mut input: ArrayD<f32>) -> Array2<f32> {
        for i in 0..self.layers.len() {
            input = self.layers[i].predict(input);
        }
        input.into_dimensionality::<Ix2>().unwrap()
    }

    /// This function calculates the inference accuracy on a testset with given labels.
    pub fn test(&self, input: ArrayD<f32>, target: Array2<f32>) {
        let n = target.len_of(Axis(0));
        let mut loss: Array1<f32> = Array1::zeros(n);
        let mut correct: Array1<f32> = Array1::ones(n);
        par_azip!((index i, l in &mut loss, c in &mut correct) {
            let current_input = input.index_axis(Axis(0), i);
            let current_fb = target.index_axis(Axis(0), i);
            let pred = self.predict(current_input.into_owned().into_dyn());
            *l = self.loss_from_prediction(pred.clone(), current_fb.into_owned());

            let best_guess: f32 = (pred.clone() * current_fb).sum();

            let num: usize = pred.iter().filter(|&x| *x >= best_guess).count();
            if num != 1 {
                *c = 0.;
            }
        });
        let avg_loss = loss.par_iter().sum::<f32>() / (n as f32);
        let acc = correct.par_iter().sum::<f32>() / (n as f32);
        println!("avg loss: {}, percentage correct: {}", avg_loss, acc);
    }

    /// This function calculates the loss based on the neural network inference and a target label.
    pub fn loss_from_prediction(&self, prediction: Array1<f32>, target: Array1<f32>) -> f32 {
        let y = prediction.into_dyn();
        let t = target.into_dyn();
        let loss = self.error_function.forward(y, t);
        loss[0]
    }

    /// This function calculates the loss based on the original data and the target label.
    pub fn loss_from_input(&self, mut input: ArrayD<f32>, target: Array1<f32>) -> f32 {
        let n = self.layers.len();
        for i in 0..(n - 1) {
            input = self.layers[i].predict(input);
        }

        let loss;
        if self.from_logits {
            loss = self
                .error_function
                .loss_from_logits(input, target.into_dyn());
        } else {
            loss = self.error_function.forward(input, target.into_dyn());
        };
        loss[0]
    }

    /// This function handles training on a single 1d example.
    pub fn train1d(&mut self, input: Array1<f32>, target: Array1<f32>) {
        self.train(input.into_dyn(), target.into_dyn());
    }
    /// This function handles training on a single 2d example.
    pub fn train2d(&mut self, input: Array2<f32>, target: Array1<f32>) {
        self.train(input.into_dyn(), target.into_dyn());
    }
    /// This function handles training on a single 3d example.
    pub fn train3d(&mut self, input: Array3<f32>, target: Array1<f32>) -> Array1<f32> {
        self.train(input.into_dyn(), target.into_dyn())
            .into_dimensionality::<Ix1>()
            .unwrap()
    }
    /// This function handles training on a single dynamic-dimensional example.
    pub fn train(&mut self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        //assert_eq!(input.len_of(Axis(0)), target.len()); //later when training on batches
        //maybe return option(accuracy,None) and add a setter to return accuracy?
        let mut input = input.into_dyn();
        let n = self.layers.len();

        // forward pass
        // handle layers 1 till pre-last
        for i in 0..(n - 1) {
            input = self.layers[i].forward(input);
        }
        let res = input.clone();

        let mut feedback;
        // handle last layer and error function
        if self.from_logits {
            //merge last layer with error function
            feedback = self
                .error_function
                .deriv_from_logits(input, target.into_dyn());
        // to print error function loss here: println!("{}", self.error_function.loss_from_logits(input, target);
        } else {
            //evaluate last activation layer and error function seperately
            input = self.layers[n - 1].forward(input);
            // to print error function loss here: println!("{}", self.error_function.loss(input, target);
            feedback = self.error_function.backward(input, target.into_dyn());
            feedback = self.layers[n - 1].backward(feedback);
        }

        // backward pass
        // handle pre-last till first layer
        for i in (0..(n - 1)).rev() {
            feedback = self.layers[i].backward(feedback);
        }
        res
    }
}
