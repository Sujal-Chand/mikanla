pub mod activation;
pub mod builder;
pub mod dataset;
pub mod error;
pub mod layers;
pub mod loss;
pub mod network;
pub mod tensor;
pub mod training;

pub mod prelude {
    pub use crate::activation::Activation;
    pub use crate::builder::NetworkBuilder;
    pub use crate::dataset::{Dataset, TrainingSample};
    pub use crate::error::NNError;
    pub use crate::layers::Layer;
    pub use crate::loss::{mse_gradient, mse_loss};
    pub use crate::network::Network;
    pub use crate::tensor::Tensor;
    pub use crate::training::{TrainingConfig, TrainingHistory};
}
