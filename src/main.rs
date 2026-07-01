use mikanla::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let network = NetworkBuilder::new()
        .input(3)
        .dense(4, Activation::ReLU)
        .dense(2, Activation::Sigmoid)
        .build();

    let input = Tensor::new(vec![1.0, 0.5, -0.25]);
    let output = network.forward(input)?;

    println!("output: {:?}", output.data());
    Ok(())
}
