use std::fmt; // imports the rust formatting module in order to implement the Display trait

// create a custom error type for the neural network library
#[derive(Debug)]
pub enum NNError {
    InputSizeMismatch { expected: usize, got: usize },
    MissingForwardPass,
    EmptyDataset,
}

// implement the Display trait for the custom error types
#[rustfmt::skip] // skip formatting as causes rust-analyzer issue on macOS VSCode
impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NNError::InputSizeMismatch { expected, got } => write!(
                f, 
                "Input size mismatch: expected {}, got {}", expected, got
            ),
            NNError::MissingForwardPass => write!(
                f,
                "Missing forward pass: the network must perform a forward pass before this operation"
            ),
            NNError::EmptyDataset => write!(f, "Dataset must contain at least one sample"),
        }
    }
}

// tells the complier that NNError is a valid error type
impl std::error::Error for NNError {}
