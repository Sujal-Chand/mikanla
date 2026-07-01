pub mod dense;

use crate::error::NNError;
use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, NNError>;
}
