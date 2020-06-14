use crate::network::nn::NeuralNetwork;
use ndarray::{array};
use rand::Rng;

pub struct Game2 {
    res: (u32, u32, u32),
    last_result: i32,
    nn: NeuralNetwork,
}

impl Game2 {
    pub fn new(_iterations: usize) -> Result<Game2, String> {
      let mut nn = NeuralNetwork::new(2);
      nn.add_connection("dense", 4); //Dense with 2 / 3 output neurons
      nn.add_activation("sigmoid"); //Sigmoid
      nn.add_connection("dense", 1); //Dense with 1 output neuron
      nn.add_activation("sigmoid"); //Sigmoid
      Ok(Game2 {
          res: (0, 0, 0),
          last_result: 42, //init value shouldn't be used
          nn,
      })
    }

    fn update_results(&mut self, first_player_fields: u8, second_player_fields: u8) {
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

    pub fn bench(&mut self, num_games: u64) -> (u32, u32, u32) {
        self.play_games(num_games, false)
    }

    fn play_games(&mut self, num_games: u64, train: bool) -> (u32, u32, u32) {
        self.res = (0, 0, 0);
        let mut counter = 0;

        let input = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.]]; // FIRST
        let fb = array![[0.],[0.],[1.],[1.]]; //First works good with 200k examples

        println!("input row 0, {}", input.nrows());
        for _ in 0..num_games {
          counter += 1;
          let move_number = rand::thread_rng().gen_range(0, input.nrows()) as usize;
          let mut current_input = input.row(move_number).into_owned().clone();
          self.nn.forward(current_input);
          if train {
            self.nn.backward(fb.row(move_number).into_owned());
          }
          if counter % 20 == 0 {
            self.nn.error();
          }
        }
        self.res
    }
}
