use rand::RngExt;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    pub fn random(total_size: usize, input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(total_size);

        let limit = (6.0 / (input_size + output_size) as f32).sqrt();

        for _ in 0..total_size {
            data.push(rng.random_range(-limit..limit));
        }
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}
