use crate::engine::engine_trait::Engine;
use crate::board::board_trait::BoardInfo;
use crate::network::nn::NeuralNetwork;
use ndarray::{Array,Array1};
use rand::Rng;

//""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

pub struct GDEngine{
  first_player: bool,
  rounds: u8,
  batch_size: u32,
  games_played: usize,
  nn: NeuralNetwork,
  results: Vec<i32>,
  move_stats: (usize,usize), //legal vs illegal
}

impl GDEngine {
    pub fn new(rounds_per_game: u8, is_first_player: bool) -> Self {
        
        GDEngine {
            first_player: is_first_player,
            rounds: rounds_per_game,
            batch_size: 1,
            games_played: 0,
            nn: NeuralNetwork::new(36),
            results: vec![],
            move_stats: (0,0),
        }
    }

}

fn random_select(move_probs: Array1<f32>) -> usize {
  let sum = move_probs.iter().fold(0.0, |sum, val| sum + val.abs());
  let mut move_pos: f32 = rand::thread_rng().gen_range(0.0, sum);
  let mut pos = 0;
  while move_pos > 0.0 {
    move_pos -= move_probs[pos].abs();
    pos += 1;
  }
  pos
}

fn expand_to_36(legal_moves: Vec<f32>) -> Array1<f32> {
  let mut expanded_arr: Array1<f32> = Array::zeros(36);
  for pos in 0..legal_moves.len() {
    expanded_arr[pos] = 1.0 ; // n; for sigmoid don't normalize, only for softmax
  }
  expanded_arr
}

impl Engine for GDEngine {
    fn get_id(&self) -> String {
        "gd engine".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      let board_position = board.get_board_position().iter().map(|&x| x as f32).collect();
      let board_position: Array1<f32> = Array::from_shape_vec(36, board_position).unwrap();
      let move_probs: Array1<f32> = self.nn.forward(board_position);

      let legal_moves: Vec<f32> = board.get_possible_moves().iter().map(|&x| x as f32).collect();
      let legal_moves: Array1<f32> = expand_to_36(legal_moves);
      let mut proposed_move: usize = random_select(move_probs);
      self.nn.backward(legal_moves);

      if !board.get_possible_moves().contains(&proposed_move) {
        let legal_moves = board.get_possible_moves();
        proposed_move = legal_moves[rand::thread_rng().gen_range(0, legal_moves.len())];
        //println!("taking over failed NN prediction {}, now {}", old_move, proposed_move);
        self.move_stats.1 += 1;
      } else {
        self.move_stats.0 += 1;
        //println!("taking NN prediction {}", proposed_move);
      }
      proposed_move
    }
    
    fn reset_board(&mut self) {
    }

    fn finish_round(&mut self, result: i32) {
      
      self.results.push(result);
      self.games_played += 1;

      if self.games_played % 10 == 0 {
        self.nn.error();
      }

      if self.games_played % 1000 == 0 { //reset stats for time after learning rules
        println!("stats for last 1k games, total games: {}", self.games_played);
        println!("llegal: {} vs illegal: {}", self.move_stats.0, self.move_stats.1);
        self.move_stats = (0,0);
      }
    }

}
