use rand::SeedableRng;
use rand::rngs::{StdRng, SysRng};

use crate::activation::Activation;
use crate::layers::Layer;
use crate::layers::dense::Dense;
use crate::network::Network;

pub struct NetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    current_size: Option<usize>,
    rng: StdRng,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            current_size: None,
            rng: StdRng::try_from_rng(&mut SysRng).unwrap(),
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    pub fn input(mut self, size: usize) -> Self {
        self.current_size = Some(size);
        self
    }

    pub fn dense(mut self, output_size: usize, activation: Activation) -> Self {
        let input_size = self
            .current_size
            .expect("Input size must be specified before adding layers.");

        let dense = Dense::new(input_size, output_size, activation, &mut self.rng);

        self.layers.push(Box::new(dense));
        self.current_size = Some(output_size);

        self
    }

    pub fn build(self) -> Network {
        Network::new(self.layers)
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}
