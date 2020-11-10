use ndarray::ArrayD;

/// An interface for all relevant functions which an error function has to implement.
///
/// This layer may only be called during a training run, where the label (ground trouth) is known.
/// The loss/driv_from_logits functions may directly forward input/output to/from forward/backward.   
/// They exist in order to allow a (numerically/performancewise/...) optimized implementation in combination  
/// with a specific activation function as the last layer of the neural network.
/// Examples are sigmoid+bce or softmax+cce.
pub trait Error: Send + Sync {
    /// This function returns a unique string identifying the type of the error function.
    fn get_type(&self) -> String;

    /// This function is used to calculate the error based on the last layer output and the expected output.
    fn forward(&self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    /// This function is used to calculate the error to update previous layers.
    fn backward(&self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    /// This function takes the output of the neural network *before* the last activation function.   
    /// It merges the functionality of the last activation function with the forward() function in an improved implementation.   
    /// Only works for a given activation function.
    fn loss_from_logits(&self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;

    /// Similare to the loss_from_logits() function it takes the output *before* the last activation function.   
    /// It merges the functionality of the backward() function with the backward() function of a specific activation function.
    fn deriv_from_logits(&self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32>;
}
