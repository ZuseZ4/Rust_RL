use hello_rust::game::game;
use hello_rust::game::xor;
use std::io;

//just for mnist test
use hello_rust::network::nn::NeuralNetwork;
use ndarray::{array, Array, Array1, Array2, Array3, Axis};
use rand::Rng;
use mnist::{Mnist, MnistBuilder};

fn test_MNIST() {
  let (trn_size, rows, cols) = (60_000, 28, 28);

  // Deconstruct the returned Mnist struct.
  let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
      .label_format_one_hot() //0..9 
      .finalize();

  // Get the label of the first digit.
  let n = 1;
  let trn_lbl = Array2::from_shape_vec((trn_size,10),trn_lbl).unwrap();
  let first_label = trn_lbl.index_axis(Axis(0),n);
  println!("The first digit is a {}.", first_label);

  // Convert the flattened training images vector to a matrix.
  let mut trn_img: Array3<f32> = Array3::from_shape_vec((trn_size,rows,cols), trn_img).unwrap().mapv(|x| x as f32);
  trn_img.mapv_inplace(|x| x/256.0);

  // Get the image of the first digit.
  let first_image = trn_img.index_axis(Axis(0),n);
  assert_eq!(first_image.shape(), &[28,28]);

  // Get the image of the first digit and round the values to the nearest tenth.
  let trn_show = trn_img.mapv(|x| (x*10.0).round()/20.0) ;//only to show
  let first_image = trn_show.index_axis(Axis(0),n);
  println!("The image looks like... \n{:#?}", first_image);
}

fn main() -> Result<(),String>{

    test_MNIST();

    // set number of rounds to play per game
    let mut engines = String::new();
    let mut rounds = String::new();
    let mut training_games = String::new();
    let mut bench_games = String::new();
    println!("please insert the number of rounds per game.");
    io::stdin()
        .read_line(&mut rounds)
        .expect("Failed to read number of rounds");

    println!("please insert the number of training games.");
    io::stdin()
        .read_line(&mut training_games)
        .expect("Failed to read number of games");

    println!("please insert the number of benchmark games.");
    io::stdin()
        .read_line(&mut bench_games)
        .expect("Failed to read number of games");

    println!("please pick engines.");
    io::stdin()
        .read_line(&mut engines)
        .expect("Failed to read type of engines");

    let rounds: u8 = rounds.trim().parse().expect("please type a number");
    let training_games: u64 = training_games.trim().parse().expect("please type a number");
    let bench_games: u64 = bench_games.trim().parse().expect("please type a number");
    let engines: u8 = engines
        .trim()
        .parse()
        .expect("please type a number (11 for random-random, 22 for ai-ai, 33 for human-human");

    println!(
        "rounds: {}, #training games: {}, #bench games: {}\n",
        rounds, training_games, bench_games
    );
     


    //let rounds = 0;
    //let training_games = 40000;
    //let bench_games = 10;
    //let mut game = xor::Game2::new(rounds)?;
    
    let mut game = game::Game::new(rounds, engines)?;
    game.train(training_games);
    let res: (u32, u32, u32) = game.bench(bench_games);

    println!("engine1 ({}): {}", game.get_engine_ids().0, res.0);
    println!("draw: {}", res.1);
    println!("engine2 ({}): {}", game.get_engine_ids().1, res.2);
    Ok(())
}
