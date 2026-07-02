pub mod dense;

use crate::error::NNError;
use crate::tensor::Tensor;

pub trait Layer {
    /// Computes the forward pass.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, NNError>;

    /// Computes the backward pass.
    fn backward(&mut self, output_gradient: &Tensor, learning_rate: f32)
    -> Result<Tensor, NNError>;
}
