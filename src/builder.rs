use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::activation::Activation;
use crate::error::NNError;
use crate::layers::Layer;
use crate::layers::dense::Dense;
use crate::network::Network;

enum LayerConfig {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Option<Activation>,
    },
}

/// Builds a [`Network`] one layer configuration at a time.
///
/// A dense layer must follow an input declaration and be followed by an
/// activation:
///
/// ```
/// use mikanla::prelude::*;
///
/// let network = NetworkBuilder::new()
///     .input(2)
///     .dense(4)
///     .activation(Activation::Tanh)
///     .build()?;
///
/// # Ok::<(), NNError>(())
/// ```
pub struct NetworkBuilder {
    layer_configs: Vec<LayerConfig>,
    current_size: Option<usize>,
    rng: StdRng,
    validation_error: Option<NNError>,
}

impl NetworkBuilder {
    /// Creates an empty network builder.
    pub fn new() -> Self {
        Self {
            layer_configs: Vec::new(),
            current_size: None,
            rng: rand::make_rng(),
            validation_error: None,
        }
    }

    /// Sets the random seed used to initialize layer weights.
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Sets the size of the network input.
    pub fn input(mut self, size: usize) -> Self {
        self.current_size = Some(size);
        self
    }

    /// Adds a pending dense layer.
    ///
    /// Call [`Self::activation`] next to finish configuring the layer.
    pub fn dense(mut self, output_size: usize) -> Self {
        let Some(input_size) = self.current_size else {
            self.record_error(NNError::DenseLayerBeforeInput);
            return self;
        };

        self.layer_configs.push(LayerConfig::Dense {
            input_size,
            output_size,
            activation: None,
        });
        self.current_size = Some(output_size);

        self
    }

    /// Assigns an activation function to the most recently added layer.
    pub fn activation(mut self, activation: Activation) -> Self {
        let layer_index = self.layer_configs.len().checked_sub(1);

        match (layer_index, self.layer_configs.last_mut()) {
            (
                Some(layer_index),
                Some(LayerConfig::Dense {
                    activation: layer_activation,
                    ..
                }),
            ) => {
                if layer_activation.is_some() {
                    self.record_error(NNError::ActivationAlreadyAssigned { layer_index });
                } else {
                    *layer_activation = Some(activation);
                }
            }
            _ => self.record_error(NNError::ActivationBeforeLayer),
        }

        self
    }

    /// Builds a network from the configured layers.
    ///
    /// # Errors
    ///
    /// Returns an [`NNError`] if a builder method was called in an invalid
    /// order or if any layer is missing its activation function.
    pub fn build(mut self) -> Result<Network, NNError> {
        if let Some(error) = self.validation_error {
            return Err(error);
        }

        let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(self.layer_configs.len());

        for (layer_index, config) in self.layer_configs.into_iter().enumerate() {
            match config {
                LayerConfig::Dense {
                    input_size,
                    output_size,
                    activation,
                } => {
                    let activation =
                        activation.ok_or(NNError::MissingActivation { layer_index })?;
                    let dense = Dense::new(input_size, output_size, activation, &mut self.rng);
                    layers.push(Box::new(dense));
                }
            }
        }

        Ok(Network::new(layers))
    }

    fn record_error(&mut self, error: NNError) {
        if self.validation_error.is_none() {
            self.validation_error = Some(error);
        }
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}
