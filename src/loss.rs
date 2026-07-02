use crate::error::NNError;
use crate::tensor::Tensor;

pub fn mse_loss(predicted: &Tensor, target: &Tensor) -> Result<f32, NNError> {
    if predicted.len() != target.len() {
        return Err(NNError::InputSizeMismatch {
            expected: target.len(),
            got: predicted.len(),
        });
    }

    let mut total = 0.0;
    for i in 0..predicted.len() {
        let error = predicted.data()[i] - target.data()[i];
        total += error * error;
    }

    Ok(total / predicted.len() as f32)
}

pub fn mse_gradient(predicted: &Tensor, target: &Tensor) -> Result<Tensor, NNError> {
    if predicted.len() != target.len() {
        return Err(NNError::InputSizeMismatch {
            expected: target.len(),
            got: predicted.len(),
        });
    }

    let mut gradient = Vec::with_capacity(predicted.len());

    for i in 0..predicted.len() {
        let error = predicted.data()[i] - target.data()[i];
        gradient.push(2.0 * error / predicted.len() as f32);
    }

    Ok(Tensor::new(gradient))
}
