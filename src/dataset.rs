use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TrainingSample {
    input: Tensor,
    target: Tensor,
}

impl TrainingSample {
    pub fn new(input: Tensor, target: Tensor) -> Self {
        Self { input, target }
    }

    pub fn from_vecs(input: Vec<f32>, target: Vec<f32>) -> Self {
        Self {
            input: Tensor::new(input),
            target: Tensor::new(target),
        }
    }

    pub fn input(&self) -> &Tensor {
        &self.input
    }

    pub fn target(&self) -> &Tensor {
        &self.target
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    samples: Vec<TrainingSample>,
}

impl Dataset {
    pub fn new(samples: Vec<TrainingSample>) -> Self {
        Self { samples }
    }

    pub fn from_pairs(pairs: Vec<(Vec<f32>, Vec<f32>)>) -> Self {
        let samples = pairs
            .into_iter()
            .map(|(input, target)| TrainingSample::from_vecs(input, target))
            .collect();

        Self { samples }
    }

    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
