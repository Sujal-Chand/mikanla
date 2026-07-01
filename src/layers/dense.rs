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
    /// Activation function applied element-wise to the outputs (e.g., ReLU, Sigmoid).
    activation: Activation,
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
            // Initializes weights randomly.
            // Note: input_size * output_size assumes a flattened 1D representation.
            weights: Tensor::random(input_size * output_size, input_size, output_size),
            // Biases are typically initialized to zero.
            biases: Tensor::zeros(output_size),
            activation,
        }
    }

    /// Helper function to calculate the 1D index of a 2D weight matrix.
    /// Maps a 2D weight coordinate (output_row, input_col) to a row-major 1D flat index.
    fn weight_index(&self, output: usize, input: usize) -> usize {
        output * self.input_size + input
    }
}

impl Layer for Dense {
    /// Computes the forward pass of the layer
    ///
    /// # Errors
    /// Returns a `NNError::InputSizeMismatch` if the input tensor's length doesn't match `input_size`.
    fn forward(&self, input: &Tensor) -> Result<Tensor, NNError> {
        // Guard clause: Validate that incoming data matches expected dimensions.
        if input.len() != self.input_size {
            return Err(NNError::InputSizeMismatch {
                expected: self.input_size,
                got: input.len(),
            });
        }

        // Pre-allocate memory for the output vector to avoid dynamic reallocations.
        let mut output = Vec::with_capacity(self.output_size);

        // Manually compute the matrix-vector multiplication and add bias: (Input * Weights) + Bias
        for out_i in 0..self.output_size {
            // Start with the bias for the current output neuron
            let mut sum = self.biases.data()[out_i];

            // Compute dot product of the input vector and the weights for this specific neuron
            for in_i in 0..self.input_size {
                let input_value = input.data()[in_i];
                let weight_value = self.weights.data()[self.weight_index(out_i, in_i)];

                sum += input_value * weight_value;
            }

            // Pass the resulting linear combination through the activation function
            output.push(self.activation.apply(sum));
        }

        // Wrap the final output vector in a Tensor and return it
        Ok(Tensor::new(output))
    }
}
