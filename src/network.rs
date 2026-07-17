use crate::error::NNError;
use crate::layers::Layer;
use crate::tensor::Tensor;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, NNError> {
        let mut layers = self.layers.iter_mut();

        let Some(first_layer) = layers.next() else {
            return Ok(input.clone());
        };

        let mut current = first_layer.forward(input)?;

        for layer in layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    pub fn backward(
        &mut self,
        output_gradient: Tensor,
        learning_rate: f32,
    ) -> Result<Tensor, NNError> {
        let mut current_gradient = output_gradient;

        for layer in self.layers.iter_mut().rev() {
            current_gradient = layer.backward(&current_gradient, learning_rate)?;
        }

        Ok(current_gradient)
    }
}
