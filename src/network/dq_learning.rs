use rand::Rng;
use rand::ThreadRng;
use std::collections::HashMap;
use crate::board::board_trait::BoardInfo;
use crate::network::nn::NeuralNetwork;

fn new() -> NeuralNetwork {
  let mut nn = NeuralNetwork::new2d((6, 6), "cce".to_string());
  nn.set_batch_size(32);
  nn.set_learning_rate(0.1);
  nn.add_convolution((3,3), 32);
  nn.add_activation("sigmoid");
  nn.add_flatten();
  //nn.add_dense(10); //Dense with 10 output neuron
  //nn.add_activation("sigmoid");
  nn.add_dense(36); //Dense with 10 output neuron
  nn.add_activation("sigmoid");
  nn
}


#[allow(dead_code)]
pub struct DQlearning {
    nn: NeuralNetwork,
    counter: usize,
    sum: usize,
    exploration: f32,
    learning_rate: f32,
    discount_factor: f32,
    scores: HashMap<(String,usize), f32>, // (State,Action), reward
    last_state: String,
    last_action: usize, //TODO change action to String for generalization
    rng: ThreadRng,
}

impl DQlearning {
    pub fn new(exploration: f32) -> Self {
      let learning_rate = 0.1f32;
      let discount_factor = 0.95f32;
        DQlearning {
            sum: 0,
            counter: 0,
            nn: new(),
            exploration,
            learning_rate,
            discount_factor,
            last_action: 42 as usize,
            last_state: "".to_string(),
            scores: HashMap::new(),
            rng: rand::thread_rng(),
        }
    }

    pub fn get_exploration_rate(&self) -> f32 {
      self.exploration
    }

    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(),String>{
      if e < 0. || e > 1. {
        return Err("exploration rate must be in [0,1]!".to_string());
      }
      self.exploration = e;
      Ok(())
    }
}


impl DQlearning {

    pub fn reset_board(&mut self) {
    }

    // update "table" based on last action and their result
    pub fn finish_round(&mut self, result: i32) { // -1 for loss, 0 for draw, 1 for win
    }
    
    pub fn get_move(&mut self, board: &impl BoardInfo) -> usize { //get reward for previous move

        let (board_string, actions, reward) = board.step(); 
        let (board_arr, action_arr) = board.step2();

        let (best_move, max_future_q) = (actions[0], 42.); 

  
        let predicted_moves = self.nn.predict2d(board_arr.clone());
        self.nn.train(board_arr.into_dyn(),action_arr.clone());
        let errors = predicted_moves.iter().zip(action_arr.iter()).fold(0, |err, val| if f32::abs(val.0-val.1) > 0.2 {err+1} else {err});
        self.sum += errors;
        self.counter += 1;
        if self.counter % 1000 == 0 {
          println!("errors per 1k moves: {}",self.sum);
          self.sum = 0;
          self.counter = 0;
        }



        // update HashMap-entry for last (state,action) based on the received reward
        //self.update_Map(reward, max_future_q);

        self.last_action = best_move;
        self.last_state = board_string;

        //if self.exploration > rand::thread_rng().gen() {
        if 1==1 {
          self.last_action = self.get_random_move(actions);
        }

        self.last_action
    }


    fn get_random_move(&mut self, actions: Vec<usize>) -> usize {
        let position = self.rng.gen_range(0, actions.len()) as usize;
        return actions[position];
    }

 
}
