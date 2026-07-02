use mikanla::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = NetworkBuilder::new()
        .input(3)
        .dense(4, Activation::ReLU)
        .dense(2, Activation::Sigmoid)
        .build();

    let input = Tensor::new(vec![1.0, 0.5, -0.25]);
    let target = Tensor::new(vec![1.0, 0.0]);
    let learning_rate = 0.01;
    let epochs = 1000;

    // output before training
    let starting_output = network.forward(input.clone())?;
    let starting_loss = mse_loss(&starting_output, &target)?;

    println!("starting output: {:?}", starting_output.data());
    println!("target goal: {:?}", target.data());
    println!("starting loss: {}\n", starting_loss);
    for epoch in 0..epochs {
        // forward pass
        let output = network.forward(input.clone())?;
        // calculate loss
        let loss = mse_loss(&output, &target)?;
        // calculate gradient compared to output
        let gradient = mse_gradient(&output, &target)?;
        // backward pass updates weights
        network.backward(gradient, learning_rate)?;
        if epoch % 100 == 0 {
            println!("epoch: {}, loss: {}", epoch, loss);
        }
    }
    let final_output = network.forward(input)?;
    println!("\nfinal output: {:?}", final_output.data());
    Ok(())
}
