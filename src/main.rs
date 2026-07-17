use mikanla::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::new()
        .input(3)
        .dense(4, Activation::ReLU)
        .dense(2, Activation::Sigmoid)
        .build();

    let input = Tensor::new(vec![1.0, 0.5, -0.25]);
    let target = Tensor::new(vec![1.0, 0.0]);

    let config = TrainingConfig::new(1000, 0.01);

    let history = network.train(&input, &target, config)?;

    let output = network.forward(&input)?;

    println!("\nstarting loss: {}", history.initial_loss());
    println!("final loss: {}", history.final_loss());
    println!("final output: {:?}", output.data());

    Ok(())
}
