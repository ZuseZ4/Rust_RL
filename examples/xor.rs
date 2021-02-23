use ndarray::{array, Array2};
use rust_rl::network::nn::NeuralNetwork;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(vec![2], "none".to_string(), "adam".to_string());
    nn.set_batch_size(4);
    nn.set_learning_rate(0.01);
    nn.add_dense(2); //Dense with 2 output neurons
    nn.add_activation("sigmoid");
    nn.add_dense(2);
    nn.add_activation("softmax");
    nn
}

fn test(nn: &mut NeuralNetwork, input: &Array2<f32>, feedback: &Array2<f32>) {
    let pred = nn.predict(input.clone().into_dyn());
    println!(
        "prediction was: \n {:} \n should have been: \n {:} \n",
        pred, feedback
    );
}

fn train(nn: &mut NeuralNetwork, num_games: usize, input: &Array2<f32>, feedback: &Array2<f32>) {
    for _ in 0..num_games {
        nn.train(input.clone().into_dyn(), feedback.clone().into_dyn());
    }
}

pub fn main() {
    let input = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]]; // XOR
    let feedback = array![[0., 1.], [1., 0.], [1., 0.], [0., 1.]]; //XOR works good with 2k examples
    let mut nn = new();
    nn.print_setup();

    for _ in 0..10 {
        train(&mut nn, 200, &input, &feedback);
        test(&mut nn, &input, &feedback);
    }
}
