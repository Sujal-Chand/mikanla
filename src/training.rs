use crate::error::NNError;
use crate::loss::{mse_gradient, mse_loss};
use crate::network::Network;
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct TrainingConfig {
    epochs: usize,
    learning_rate: f32,
    log_every: Option<usize>,
}

impl TrainingConfig {}
