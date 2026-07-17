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

impl TrainingConfig {
    pub fn new(epochs: usize, learning_rate: f32) -> Self {
        Self {
            epochs,
            learning_rate,
            log_every: None,
        }
    }

    pub fn log_every(mut self, interval: usize) -> Self {
        self.log_every = Some(interval.max(1));
        self
    }
}

#[derive(Debug, Clone)]
pub struct TrainingHistory {
    losses: Vec<f32>,
    initial_loss: f32,
    final_loss: f32,
}

impl TrainingHistory {
    pub fn losses(&self) -> &[f32] {
        &self.losses
    }

    pub fn initial_loss(&self) -> f32 {
        self.initial_loss
    }

    pub fn final_loss(&self) -> f32 {
        self.final_loss
    }
}

impl Network {
    pub fn train(
        &mut self,
        input: &Tensor,
        target: &Tensor,
        config: TrainingConfig,
    ) -> Result<TrainingHistory, NNError> {
        let starting_output = self.forward(input)?;
        let initial_loss = mse_loss(&starting_output, target)?;

        let mut losses = Vec::with_capacity(config.epochs);

        for epoch in 0..config.epochs {
            // compute prediction
            let output = self.forward(input)?;

            // calculate loss
            let loss = mse_loss(&output, target)?;

            // calculate output gradient
            let gradient = mse_gradient(&output, target)?;

            // propagate gradient backwards and update weights
            self.backward(gradient, config.learning_rate)?;

            losses.push(loss);

            if let Some(interval) = config.log_every {
                if epoch % interval == 0 {
                    println!("epoch: {epoch}, loss: {loss}");
                }
            }
        }

        let final_output = self.forward(input)?;
        let final_loss = mse_loss(&final_output, target)?;

        Ok(TrainingHistory {
            losses,
            initial_loss,
            final_loss,
        })
    }
}
