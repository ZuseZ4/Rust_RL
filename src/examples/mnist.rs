use crate::network::nn::NeuralNetwork;
use ndarray::{Array2, Array3, Axis};
use rand::Rng;
use mnist::{Mnist, MnistBuilder};

fn new() -> NeuralNetwork {
  let mut nn = NeuralNetwork::new2d((28, 28), "cce".to_string());
  nn.set_batch_size(32);
  nn.set_learning_rate(0.0002);
  nn.add_flatten();
  //nn.add_dense(128); //Dense with 32 output neurons
  //nn.add_activation("sigmoid"); //Sigmoid
  nn.add_dense(10); //Dense with 10 output neuron
  nn.add_activation("softmax"); //Softmax
  nn
}

fn test(mut nn: NeuralNetwork, input: Array3<f32>, feedback: Array2<f32>) {
  let mut counter = 0;
  for (current_input, current_fb) in input.outer_iter().zip(feedback.outer_iter()) {
    let pred = &nn.forward2d(current_input.into_owned());
    let _diff: f32 = (pred - &current_fb)
      .mapv(|x| x.powi(2))
      .sum()
      / 10.; // calculating Mean Square Error (MSE)
    if counter % 100 == 0 {
      //println!("loss: {}", _diff);
      println!("prediction: {} fb: {}", pred, current_fb);
    }
    counter += 1;
  }
}

fn train(nn: &mut NeuralNetwork, num: usize, input: &Array3<f32>, fb: &Array2<f32>) {
  let mut counter = 0;
  for _ in 0..num {
    let pos = rand::thread_rng().gen_range(0, input.shape()[0]) as usize;
    let current_input = input.index_axis(Axis(0),pos).into_owned();
    let current_fb = fb.index_axis(Axis(0),pos).into_owned();
    nn.forward2d(current_input);
    nn.backward(current_fb);
    if counter % 1000 == 0 {
      println!("counter: {}", counter);
    }
    counter += 1;
  }
}


#[allow(non_snake_case)]
pub fn test_MNIST() {
  let (train_size, test_size, rows, cols) = (60_000, 10_000, 28, 28);

  // Deconstruct the returned Mnist struct.
  let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
      .label_format_one_hot() //0..9 
      .finalize();

  // changing mnist train dataset from long u8 vectors to f32 matrices
  let train_lbl = Array2::from_shape_vec((train_size,10),trn_lbl).unwrap().mapv(|x| x as f32);
  let test_lbl  = Array2::from_shape_vec((test_size,10),tst_lbl).unwrap().mapv(|x| x as f32);
  let mut train_img: Array3<f32> = Array3::from_shape_vec((train_size,rows,cols), trn_img).unwrap().mapv(|x| x as f32);
  let mut test_img:  Array3<f32> = Array3::from_shape_vec((test_size,rows,cols), tst_img).unwrap().mapv(|x| x as f32);
  assert_eq!(train_img.shape(), &[60_000,28,28]);
  assert_eq!(test_img.shape(), &[10_000,28,28]);
  println!("mapping image values from [0,255] to [0,1]");
  train_img.mapv_inplace(|x| x/256.0);
  test_img.mapv_inplace(|x| x/256.0);


  
  // Get the label of the first digit.
  let n = 1;
  let first_label = train_lbl.index_axis(Axis(0),n);
  println!("The first digit is a {}.", first_label);


  // Get the image of the first digit.
  let first_image = train_img.index_axis(Axis(0),n);
  assert_eq!(first_image.shape(), &[28,28]);

  // Get the image of the first digit and round the values to the nearest tenth. Show it.
  //let train_show = train_img.mapv(|x| (x*10.0).round()/10.0) ;//only to show
  //let first_image = train_show.index_axis(Axis(0),n);
  //println!("The image looks like... \n{:#?}", first_image);


  let mut nn = new();
  train(&mut nn, 600, &train_img, &train_lbl);
  test(nn, test_img, test_lbl);

}
