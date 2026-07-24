use mikanla::layers::dense::Dense;
use mikanla::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn builds_dense_layers_with_separate_activations() -> Result<(), NNError> {
    let mut network = NetworkBuilder::new()
        .seed(42)
        .input(2)
        .dense(4)
        .activation(Activation::Tanh)
        .dense(1)
        .activation(Activation::Sigmoid)
        .build()?;

    let output = network.forward(&Tensor::new(vec![0.0, 1.0]))?;

    assert_eq!(output.len(), 1);
    Ok(())
}

#[test]
fn separate_activation_matches_direct_dense_construction() -> Result<(), NNError> {
    let mut built_network = NetworkBuilder::new()
        .seed(42)
        .input(2)
        .dense(4)
        .activation(Activation::Tanh)
        .dense(1)
        .activation(Activation::Sigmoid)
        .build()?;

    let mut rng = StdRng::seed_from_u64(42);
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Dense::new(2, 4, Activation::Tanh, &mut rng)),
        Box::new(Dense::new(4, 1, Activation::Sigmoid, &mut rng)),
    ];
    let mut direct_network = Network::new(layers);

    let input = Tensor::new(vec![0.25, 0.75]);

    assert_eq!(
        built_network.forward(&input)?.data(),
        direct_network.forward(&input)?.data()
    );
    Ok(())
}

#[test]
fn rejects_activation_before_a_layer() {
    let result = NetworkBuilder::new().activation(Activation::Tanh).build();

    assert!(matches!(result, Err(NNError::ActivationBeforeLayer)));
}

#[test]
fn rejects_a_layer_without_an_activation() {
    let result = NetworkBuilder::new().input(2).dense(4).build();

    assert!(matches!(
        result,
        Err(NNError::MissingActivation { layer_index: 0 })
    ));
}

#[test]
fn rejects_two_activations_for_the_same_layer() {
    let result = NetworkBuilder::new()
        .input(2)
        .dense(4)
        .activation(Activation::Tanh)
        .activation(Activation::Sigmoid)
        .build();

    assert!(matches!(
        result,
        Err(NNError::ActivationAlreadyAssigned { layer_index: 0 })
    ));
}

#[test]
fn rejects_a_dense_layer_before_the_input() {
    let result = NetworkBuilder::new()
        .dense(4)
        .activation(Activation::Tanh)
        .build();

    assert!(matches!(result, Err(NNError::DenseLayerBeforeInput)));
}

#[test]
fn xor_network_still_trains() -> Result<(), NNError> {
    let mut network = NetworkBuilder::new()
        .seed(42)
        .input(2)
        .dense(4)
        .activation(Activation::Tanh)
        .dense(1)
        .activation(Activation::Sigmoid)
        .build()?;

    let dataset = Dataset::from_pairs(vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]);

    let history = network.train_dataset(&dataset, TrainingConfig::new(50_000, 0.01))?;

    assert!(history.final_loss() < history.initial_loss());

    for sample in dataset.samples() {
        let prediction = network.forward(sample.input())?;
        let predicted_class = prediction.get(0) >= 0.5;
        let expected_class = sample.target().get(0) >= 0.5;
        assert_eq!(predicted_class, expected_class);
    }

    Ok(())
}
