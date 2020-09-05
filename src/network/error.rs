use crate::network::concrete_error;

use concrete_error::bce::BinaryCrossEntropyError;
use concrete_error::cce::CategoricalCrossEntropyError;
use concrete_error::noop::NoopError;

use crate::network::error_trait::Error;
use ndarray::ArrayD;

pub enum ErrorType {
    BCE(BinaryCrossEntropyError),
    CCE(CategoricalCrossEntropyError),
    N(NoopError),
}

impl ErrorType {
    pub fn new_noop() -> ErrorType {
        ErrorType::N(NoopError::new())
    }

    pub fn new_error(error_type: String) -> Result<ErrorType, String> {
        match error_type.as_str() {
            "bce" => Ok(ErrorType::BCE(BinaryCrossEntropyError::new())),
            "cce" => Ok(ErrorType::CCE(CategoricalCrossEntropyError::new())),
            "noop" => Ok(ErrorType::N(NoopError::new())),
            _ => Err(format!("Unknown Error Function: {}", error_type)),
        }
    }
}

impl Error for ErrorType {
    fn get_type(&self) -> String {
        match self {
            ErrorType::BCE(bce_error) => bce_error.get_type(),
            ErrorType::CCE(cce_error) => cce_error.get_type(),
            ErrorType::N(noop_error) => noop_error.get_type(),
        }
    }
    fn forward(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            ErrorType::BCE(bce_error) => bce_error.forward(input, feedback),
            ErrorType::CCE(cce_error) => cce_error.forward(input, feedback),
            ErrorType::N(noop_error) => noop_error.forward(input, feedback),
        }
    }
    fn backward(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            ErrorType::BCE(bce_error) => bce_error.backward(input, feedback),
            ErrorType::CCE(cce_error) => cce_error.backward(input, feedback),
            ErrorType::N(noop_error) => noop_error.backward(input, feedback),
        }
    }
    fn loss_from_logits(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            ErrorType::BCE(bce_error) => bce_error.loss_from_logits(input, feedback),
            ErrorType::CCE(cce_error) => cce_error.loss_from_logits(input, feedback),
            ErrorType::N(noop_error) => noop_error.loss_from_logits(input, feedback),
        }
    }
    fn deriv_from_logits(&mut self, input: ArrayD<f32>, feedback: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            ErrorType::BCE(bce_error) => bce_error.deriv_from_logits(input, feedback),
            ErrorType::CCE(cce_error) => cce_error.deriv_from_logits(input, feedback),
            ErrorType::N(noop_error) => noop_error.deriv_from_logits(input, feedback),
        }
    }
}
