use mikanla::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // network definition with fixed seed to reproduce same results
    let mut network = NetworkBuilder::new()
        .seed(42)
        .input(2)
        .dense(4, Activation::Tanh)
        .dense(1, Activation::Sigmoid)
        .build();

    // the dataset that mikanla has to learn (XOR)
    let dataset = Dataset::from_pairs(vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]);

    // network training configuration
    let training_config = TrainingConfig::new(50_000, 0.01);

    // network history stats
    let history = network.train_dataset(&dataset, training_config)?;
    println!("starting loss: {}", history.initial_loss());
    println!("final loss: {}", history.final_loss());

    // testing the trained models outputs
    for sample in dataset.samples() {
        let output = network.forward(sample.input())?;
        println!("{:?} -> {:?}", sample.input().data(), output.data());
    }

    // we f**king did it!
    Ok(())
}
