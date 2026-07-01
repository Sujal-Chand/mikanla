pub mod activation;
pub mod builder;
pub mod error;
pub mod layers;
pub mod network;
pub mod tensor;

pub mod prelude {
    pub use crate::activation::Activation;
    pub use crate::builder::NetworkBuilder;
    pub use crate::error::NNError;
    pub use crate::layers::Layer;
    pub use crate::network::Network;
    pub use crate::tensor::Tensor;
}
