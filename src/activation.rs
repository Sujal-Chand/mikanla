/// Supported mathematical activation functions for the neural network layers.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// No-op activation function. Passes the input through unchanged.
    Linear,
    /// Rectified Linear Unit. Thresholds negative values to zero.
    ReLU,
    /// Sigmoid logistic function. Squashes values between 0 and 1.
    Sigmoid,
    /// Hyperbolic Tangent function. Squashes values between -1 and 1.
    Tanh,
}

impl Activation {
    /// Applies the activation function to a single scalar value.
    /// Optimized with `#[inline(always)]` to suggest the compiler replace the function
    /// call with its literal body for raw performance inside hot loops.
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
