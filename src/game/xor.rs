use crate::network::nn::NeuralNetwork;
use ndarray::{array, Array2};
use rand::Rng;

pub struct Game {
    res: (u32, u32, u32),
    nn: NeuralNetwork,
}

impl Game {
    pub fn new(_iterations: usize) -> Result<Game, String> {
      let mut nn = NeuralNetwork::new1d(2,"bce".to_string());
      nn.add_dense(2); //Dense with 2 output neurons
      nn.add_activation("sigmoid"); //Sigmoid
      nn.add_dense(1); //Dense with 1 output neuron
      nn.add_activation("sigmoid"); //Sigmoid
      nn.print_setup();
      Ok(Game {
          res: (0, 0, 0),
          nn,
      })
    }

    pub fn get_results(&self) -> (u32, u32, u32) {
        self.res
    }

    pub fn get_engine_ids(&self) -> (String, String) {
        ("Foo".to_string(),"Bar".to_string())
    }

    pub fn train(&mut self, num_games: u64) {
        self.play_games(num_games, true);
    }

    pub fn bench(&mut self, _num_games: u64) -> (u32, u32, u32) {
        self.res = (0, 0, 0);
        let input: Array2<f32> = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.]]; // FIRST
        let feedback: Array2<f32> = array![[0.],[1.],[1.],[0.]]; //First works good with 200k examples
        for i in 0..4 {
          println!("input: {}, feedback: {}", input.row(i),feedback.row(i));
          println!("output: {}, error: {}\n",self.nn.forward1d(input.row(i).into_owned()),self.nn.error(feedback.row(i).into_owned()));


        }
        self.res
    }

    fn play_games(&mut self, num_games: u64, train: bool) -> (u32, u32, u32) {
        self.res = (0, 0, 0);
        let mut counter = 0;

        let input = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.]]; // XOR
        let feedback = array![[0.],[1.],[1.],[0.]]; //First works good with 200k examples

        println!("input row 0, {}", input.nrows());
        for _ in 0..num_games {
          counter += 1;
          let move_number = rand::thread_rng().gen_range(0, input.nrows()) as usize;
          let current_input = input.row(move_number).into_owned().clone();
          self.nn.forward1d(current_input);
          if train {
            self.nn.backward(feedback.row(move_number).into_owned());
          }
          if counter % 20 == 0 {
            let current_feedback = feedback.row(move_number).into_owned();
            self.nn.error(current_feedback);
          }
        }
        self.res
    }
}
