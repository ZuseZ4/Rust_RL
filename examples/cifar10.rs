use datasets::cifar10;
use ndarray::{Array2, Array4, Axis};
use rand::Rng;
use rust_rl::network::nn::NeuralNetwork;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((3, 32, 32), "cce".to_string(), "none".to_string());
    nn.set_batch_size(32);
    nn.set_learning_rate(0.1);
    nn.add_convolution((3, 3), 10, 1);
    nn.add_activation("sigmoid");
    nn.add_dropout(0.);
    nn.add_flatten();
    nn.add_dense(10); //Dense with 10 output neuron
    nn.add_activation("softmax");
    nn
}

fn test(nn: &mut NeuralNetwork, input: &Array4<f32>, feedback: &Array2<f32>) {
    nn.test(input.clone().into_dyn(), feedback.clone());
}

fn train(nn: &mut NeuralNetwork, num: usize, input: &Array4<f32>, fb: &Array2<f32>) {
    let mut rng = rand::thread_rng();
    for _ in 0..num {
        let pos = rng.gen_range(0, input.shape()[0]) as usize;
        let current_input = input.index_axis(Axis(0), pos).into_owned();
        let current_fb = fb.index_axis(Axis(0), pos).into_owned();
        nn.train3d(current_input, current_fb);
    }
}

pub fn main() {
    let (train_size, test_size, depth, rows, cols) = (50_000, 10_000, 3, 32, 32);

    #[cfg(feature = "download")]
    cifar10::download_and_extract();
    let cifar10::Data {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = cifar10::new_normalized();
    assert_eq!(trn_img.shape(), &[train_size, depth, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, depth, rows, cols]);

    // Get the image of the first digit.
    let first_image = trn_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[3, 32, 32]);

    let mut nn = new();
    nn.print_setup();
    for i in 0..10 {
        print!("{}: ", i + 1);
        train(&mut nn, train_size, &trn_img, &trn_lbl);
        test(&mut nn, &tst_img, &tst_lbl);
    }
}
