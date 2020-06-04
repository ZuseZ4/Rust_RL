use crate::engine::engine_trait::Engine;
use crate::board::board_trait::BoardInfo;
use crate::network::nn::NeuralNetwork;
use ndarray::{Array,Array1};

//""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

pub struct GDEngine{
  first_player: bool,
  rounds: u8,
  batch_size: u32,
  games_played: usize,
  nn: NeuralNetwork,
  results: Vec<i32>,
}

impl GDEngine {
    pub fn new(rounds_per_game: u8, is_first_player: bool) -> GDEngine {
        GDEngine {
            first_player: is_first_player,
            rounds: rounds_per_game,
            batch_size: 1,
            games_played: 0,
            results: vec![],
            nn: NeuralNetwork::new(36),
        }
    }

}

fn random_select(move_probs: Array1<f32>) -> usize {
  let sum = move_probs.iter().fold(0.0, |sum, val| sum + val);
  1 as usize
}

impl Engine for GDEngine {
    fn get_id(&self) -> String {
        "gd engine".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      let board_position = board.get_board_position().iter().map(|&x| x as f32).collect();
      let board_position: Array1<f32> = Array::from_shape_vec(36, board_position).unwrap();
      let move_probs: Array1<f32> = self.nn.forward(board_position);

      let legal_moves = board.get_possible_moves().iter().map(|&x| x as f32).collect();
      let legal_moves: Array1<f32> = Array::from_shape_vec(36, legal_moves).unwrap();
      let proposed_move: usize = random_select(move_probs);
      self.nn.backward(legal_moves);
      //if !board.get_possible_moves().contains(proposed_move) {
      //  self.nn.penalize_illegal_move(normalized_move_probs);
      //  //propsed_move = // random select legal move
      //}
      proposed_move
    }
    
    fn reset_board(&mut self) {
    }

    fn finish_round(&mut self, result: i32) {
      self.results.push(result);
      self.games_played += 1;
    }

}
