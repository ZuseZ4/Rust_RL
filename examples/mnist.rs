use datasets::mnist;
use ndarray::{Array2, Array3, Axis};
use rand::Rng;
use rust_rl::network::nn::NeuralNetwork;
use std::time::Instant;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new2d((28, 28), "cce".to_string(), "adam".to_string());
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

fn test(nn: &mut NeuralNetwork, input: &Array3<f32>, feedback: &Array2<f32>) {
    nn.test(input.clone().into_dyn(), feedback.clone());
}

fn train(nn: &mut NeuralNetwork, num: usize, input: &Array3<f32>, fb: &Array2<f32>) {
    for _ in 0..num {
        let pos = rand::thread_rng().gen_range(0, input.shape()[0]) as usize;
        let current_input = input.index_axis(Axis(0), pos).into_owned();
        let current_fb = fb.index_axis(Axis(0), pos).into_owned();
        nn.train2d(current_input, current_fb);
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
    assert_eq!(trn_img.shape(), &[train_size, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, rows, cols]);

    // Get the image of the first digit.
    let first_image = trn_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[28, 28]);

    let mut nn = new();
    nn.print_setup();
    let start = Instant::now();
    for i in 0..5 {
        print!("{}: ", i + 1);
        train(&mut nn, 60_000, &trn_img, &trn_lbl); //60_000
        test(&mut nn, &tst_img, &tst_lbl);
    }
    println!("Trained for {} seconds.", start.elapsed().as_secs());
}
