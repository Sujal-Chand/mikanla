use crate::activation::Activation;
use crate::error::NNError;
use crate::layers::Layer;
use crate::tensor::Tensor;

/// Represents a Fully Connected (Dense) Layer in a neural network.
/// Each neuron in this layer is connected to every neuron in the previous layer.
pub struct Dense {
    /// Number of features in the input tensor.
    input_size: usize,
    /// Number of neurons in this layer.
    output_size: usize,
    /// Weights matrix flattened into a single Tensor.
    weights: Tensor,
    /// Bias vector for each output neuron.
    biases: Tensor,
    /// Activation function applied  to the outputs.
    activation: Activation,
    /// Cached input from the previous forward pass.
    last_input: Option<Tensor>,
    /// Cached output from the previous forward pass.
    last_output: Option<Tensor>,
}

impl Dense {
    /// Creates a new `Dense` layer with randomized weights and zeroed biases.
    ///
    /// # Arguments
    /// * `input_size` - The expected size of the incoming data vector.
    /// * `output_size` - The number of neurons/outputs this layer will produce.
    /// * `activation` - The non-linear activation function to use.
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Self {
            input_size,
            output_size,
            // Initialises weights randomly and biases to zero.
            // Note: assumes a flattened 1D representation.
            weights: Tensor::random(input_size * output_size, input_size, output_size),
            biases: Tensor::zeros(output_size),
            activation,
            last_input: None,
            last_output: None,
        }
    }
}

impl Layer for Dense {
    /// Computes the forward pass of the layer
    ///
    /// # Errors
    /// Returns a `NNError::InputSizeMismatch` if the input tensor's length doesn't match `input_size`.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, NNError> {
        // Guard clause: Validate that incoming data matches expected dimensions.
        if input.len() != self.input_size {
            return Err(NNError::InputSizeMismatch {
                expected: self.input_size,
                got: input.len(),
            });
        }

        let input_data = input.data();
        let weights_data = self.weights.data();
        let biases_data = self.biases.data();

        // Pre-allocate memory for the output vector to avoid dynamic reallocations.
        let mut output = Vec::with_capacity(self.output_size);

        // Manually compute the matrix-vector multiplication and add bias: (Input * Weights) + Bias
        for out_i in 0..self.output_size {
            let row_offset = out_i * self.input_size;
            let mut sum = biases_data[out_i];

            for in_i in 0..self.input_size {
                sum += input_data[in_i] * weights_data[row_offset + in_i];
            }

            // Pass the resulting linear combination through the activation function
            output.push(self.activation.apply(sum));
        }

        // Wrap the final output vector in a Tensor and return it
        let output = Tensor::new(output);

        self.last_input = Some(input.clone());
        self.last_output = Some(output.clone());

        Ok(output)
    }

    fn backward(
        &mut self,
        output_gradient: &Tensor,
        learning_rate: f32,
    ) -> Result<Tensor, NNError> {
        // Validate gradient size
        if output_gradient.len() != self.output_size {
            return Err(NNError::InputSizeMismatch {
                expected: self.output_size,
                got: output_gradient.len(),
            });
        }

        let input = self
            .last_input
            .as_ref()
            .ok_or(NNError::MissingForwardPass)?;

        let output = self
            .last_output
            .as_ref()
            .ok_or(NNError::MissingForwardPass)?;

        let input_size = self.input_size;
        let output_size = self.output_size;
        let activation = self.activation;

        let input_data = input.data();
        let output_data = output.data();
        let output_gradient_data = output_gradient.data();

        let mut input_gradient = vec![0.0; input_size];

        let weights_data = self.weights.data_mut();
        let biases_data = self.biases.data_mut();

        for out_i in 0..output_size {
            let activation_derivative = activation.derivative_from_output(output_data[out_i]);
            let activation_gradient = output_gradient_data[out_i] * activation_derivative;

            let row_offset = out_i * input_size;

            for in_i in 0..input_size {
                let weight_index = row_offset + in_i;

                // Use the old weight for the input gradient before updating it
                let old_weight = weights_data[weight_index];
                input_gradient[in_i] += activation_gradient * old_weight;

                let weight_gradient = activation_gradient * input_data[in_i];
                weights_data[weight_index] -= learning_rate * weight_gradient;
            }

            biases_data[out_i] -= learning_rate * activation_gradient;
        }

        Ok(Tensor::new(input_gradient))
    }
}
