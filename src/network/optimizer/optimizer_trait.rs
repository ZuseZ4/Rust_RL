use ndarray::{Array1, Array2, Array3, ArrayD};

/// A trait defining functions to alter the weight and bias updates before they are applied.
///
/// All neural network layers are expected to call the coresponding functions after calculating the deltas   
/// and only to apply the results of these functions to update their weights or biases.
pub trait Optimizer: Send + Sync {
    ///
    fn set_input_shape(&mut self, shape: Vec<usize>);
    /// Returns a string identifying the specific optimizer type. Examples are "Adam", "Momentum", and "None" for basic sgd.
    fn get_type(&self) -> String;
    /// Applies the specific optimization to dynamically shaped arrays.
    fn optimize(&mut self, weight_update: ArrayD<f32>) -> ArrayD<f32>;
    /// A wrapper around optimize().
    fn optimize1d(&mut self, weight_update: Array1<f32>) -> Array1<f32>;
    /// A wrapper around optimize()
    fn optimize2d(&mut self, weight_update: Array2<f32>) -> Array2<f32>;
    /// A wrapper around optimize()
    fn optimize3d(&mut self, weight_update: Array3<f32>) -> Array3<f32>;

    /// Allows each layer to create a copy of the given optimizer in case that he has more than one array to update.
    ///
    /// Should be called before calling any of the optimizeX functions.
    fn clone_box(&self) -> Box<dyn Optimizer>;
}
