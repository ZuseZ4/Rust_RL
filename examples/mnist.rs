use datasets::mnist;
use ndarray::{Array2, Array4, Axis};
use rand::Rng;
use rust_rl::network::nn::NeuralNetwork;
use std::time::Instant;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((1, 28, 28), "cce".to_string(), "adam".to_string());
    nn.set_batch_size(32);
    nn.set_learning_rate(1e-3);
    nn.add_convolution((3, 3), 10, 1);
    nn.add_flatten();
    nn.add_activation("sigmoid");
    nn.add_dropout(0.5);
    nn.add_dense(10);
    nn.add_activation("softmax");
    nn
}

fn test(nn: &mut NeuralNetwork, input: &Array4<f32>, feedback: &Array2<f32>) {
    nn.test(input.clone().into_dyn(), feedback.clone());
}

fn train(nn: &mut NeuralNetwork, num: usize, input: &Array4<f32>, fb: &Array2<f32>) {
    for _ in 0..num {
        let pos = rand::thread_rng().gen_range(0..input.shape()[0]) as usize;
        let current_input = input.index_axis(Axis(0), pos).into_owned();
        let current_fb = fb.index_axis(Axis(0), pos).into_owned();
        nn.train3d(current_input, current_fb);
    }
}

pub fn main() {
    let (train_size, test_size, rows, cols) = (60_000, 10_000, 28, 28);

    #[cfg(feature = "download")]
    mnist::download_and_extract();
    let mnist::Data {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = mnist::new_normalized();
    let trn_img = trn_img.into_shape((train_size, 1, rows, cols)).unwrap();
    let tst_img = tst_img.into_shape((test_size, 1, rows, cols)).unwrap();

    assert_eq!(trn_img.shape(), &[train_size, 1, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, 1, rows, cols]);

    // Get the image of the first digit.
    let first_image = trn_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[1, 28, 28]);

    let mut nn = new();
    nn.print_setup();
    train(&mut nn, 60_000, &trn_img, &trn_lbl); //60_000
    let start = Instant::now();
    for i in 0..20 {
        print!("{}: ", i + 1);
        test(&mut nn, &tst_img, &tst_lbl);
    }
    let stop = Instant::now();
    let duration = stop.duration_since(start);
    println!(
        "Trained for {},{} seconds.",
        duration.as_secs(),
        duration.subsec_millis()
    );
}
