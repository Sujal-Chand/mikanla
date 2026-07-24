# mikanla

mikanla is an experimental neural network library written from scratch in Rust.

The project is primarily a learning exercise focused on understanding how neural networks work internally, rather than relying on an existing machine learning framework. It explores the implementation of common neural network components such as tensors, dense layers, activation functions, forward propagation, backpropagation, loss functions, and training loops.

mikanla is part of a larger project called **MikaSound**, which aims to explore machine learning and audio-related applications.

## Project Goals

The main goals of mikanla are to:

* Learn how neural networks operate at a lower level
* Implement neural network components from scratch
* Experiment with different network architectures and training techniques
* Build a simple and understandable Rust API
* Explore performance improvements and parallel processing
* Eventually support models used by MikaSound
* Experiment with WebAssembly for browser-based demonstrations

## Current Features

mikanla currently includes:

* A custom tensor type
* Fully connected dense layers
* Forward propagation
* Backpropagation
* Configurable learning rates and training epochs
* Mean squared error loss
* Training history and loss tracking
* Seeded random weight initialization
* A network builder API
* Custom error handling
* Multiple activation functions:

  * Linear
  * ReLU
  * Leaky ReLU
  * Sigmoid
  * Tanh

## Example

```rust
use mikanla::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // network definition with fixed seed to reproduce the same results
    let mut network = NetworkBuilder::new()
        .seed(42)
        .input(2)
        .dense(4)
        .activation(Activation::Tanh)
        .dense(1)
        .activation(Activation::Sigmoid)
        .build()?;

    // the dataset that mikanla has to learn
    let dataset = Dataset::from_pairs(vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]);

    // network training configuration
    let training_config = TrainingConfig::new(50_000, 0.01);

    // network training history
    let history = network.train_dataset(&dataset, training_config)?;
    println!("starting loss: {}", history.initial_loss());
    println!("final loss: {}", history.final_loss());

    // test the trained model
    for sample in dataset.samples() {
        let output = network.forward(sample.input())?;
        println!("{:?} -> {:?}", sample.input().data(), output.data());
    }

    Ok(())
}
```

This example creates and trains a small neural network to learn the XOR function.

## Project Structure

```text
src/
├── activation.rs
├── builder.rs
├── dataset.rs
├── error.rs
├── layers/
│   ├── dense.rs
│   └── mod.rs
├── lib.rs
├── network.rs
├── prelude.rs
├── tensor.rs
└── training.rs
```

## Planned Features

Future experiments may include:

* Additional loss functions
* Softmax activation
* Cross-entropy loss
* Mini-batch training
* Optimizers such as momentum and Adam
* Model serialization and loading
* Convolutional layers
* Improved tensor operations
* Parallel CPU processing
* GPU acceleration
* MNIST digit classification
* WebAssembly browser demonstrations
* Audio-focused models for MikaSound

## Why Build a Neural Network Library From Scratch?

Modern machine learning frameworks handle most of the underlying mathematics and implementation details automatically. While this is useful for building real applications, it can make it difficult to understand what is happening internally.

mikanla takes the opposite approach. Its purpose is to manually implement the core operations involved in training a neural network so that concepts such as gradients, weight updates, activation derivatives, and backpropagation can be explored directly.

The library is experimental and is not intended to replace production machine learning frameworks.

## Relationship to MikaSound

**MikaSound** is the larger project that mikanla belongs to.

mikanla acts as a space for experimenting with neural network architecture, training systems, performance, and model implementation. Knowledge and components developed through mikanla may later be used to support machine learning and audio-processing features in MikaSound.

## Status

mikanla is currently under active development.

The API, project structure, and implementation may change as new concepts are explored and the library becomes more capable.

## Disclaimer

This project is being developed for learning and experimentation. It is not currently intended for production machine learning workloads.
