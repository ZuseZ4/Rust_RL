use crate::network::nn::NeuralNetwork;
use cifar_10::*;
use ndarray::{Array2, Array4, Axis};
use rand::Rng;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((3, 32, 32), "cce".to_string());
    nn.set_batch_size(32);
    nn.set_learning_rate(0.1);
    nn.add_convolution((3, 3), 16, 1);
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
    for _ in 0..num {
        let pos = rand::thread_rng().gen_range(0, input.shape()[0]) as usize;
        let current_input = input.index_axis(Axis(0), pos).into_owned();
        let current_fb = fb.index_axis(Axis(0), pos).into_owned();
        nn.train3d(current_input, current_fb);
    }
}

#[allow(non_snake_case)]
pub fn test_Cifar10() {
    let (train_size, test_size, depth, rows, cols) = (50_000, 10_000, 3, 32, 32);

    // Deconstruct the returned Cifar struct.
    let (trn_img, trn_lbl, tst_img, tst_lbl) = Cifar10::default()
        .show_images(true)
        .build()
        .expect("Failed to build CIFAR-10 data");

    // changing mnist train dataset from long u8 vectors to f32 matrices
    let train_lbl: Array2<f32> = trn_lbl.mapv(|x| x as f32);
    let test_lbl: Array2<f32> = tst_lbl.mapv(|x| x as f32);
    let mut train_img: Array4<f32> = trn_img.mapv(|x| x as f32);
    let mut test_img: Array4<f32> = tst_img.mapv(|x| x as f32);
    assert_eq!(train_img.shape(), &[train_size, depth, rows, cols]);
    assert_eq!(test_img.shape(), &[test_size, depth, rows, cols]);
    println!("mapping image values from [0,255] to [0,1]");
    train_img.mapv_inplace(|x| x / 256.0);
    test_img.mapv_inplace(|x| x / 256.0);

    // Get the image of the first digit.
    let first_image = train_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[3, 32, 32]);

    let mut nn = new();
    nn.print_setup();
    for i in 0..10 {
        println!("{}", i);
        train(&mut nn, train_size, &train_img, &train_lbl);
        test(&mut nn, &test_img, &test_lbl);
    }
}
