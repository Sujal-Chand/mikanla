/// Supported mathematical activation functions for the neural network layers.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Applies the activation function to a single scalar value.
    #[inline(always)]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
        }
    }

    /// Returns the derivative using the activated output value.
    #[inline(always)]
    pub fn derivative_from_output(self, activated_output: f32) -> f32 {
        match self {
            Activation::Linear => 1.0,
            Activation::ReLU => {
                if activated_output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => activated_output * (1.0 - activated_output),
            Activation::Tanh => 1.0 - activated_output.powi(2),
        }
    }

    /// Applies the activation function in-place across a mutable slice of data.
    /// This avoids memory allocation overhead by mutating the incoming tensor data directly.
    pub fn apply_batch(self, data: &mut [f32]) {
        match self {
            Activation::Linear => {}
            Activation::ReLU => {
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                }
            }
            Activation::Sigmoid => {
                for x in data.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
            }
            Activation::Tanh => {
                for x in data.iter_mut() {
                    *x = x.tanh();
                }
            }
        }
    }
}
