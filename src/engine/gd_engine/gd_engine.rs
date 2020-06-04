use crate::engine::engine_trait::Engine;
use crate::board::board_trait::BoardInfo;

//""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

pub struct GDEngine{
  first_player: bool,
  rounds: u8,
  batch_size: u32,
  games_played: usize,
  neural_network: NeuralNetwork,
  total_game_moves: Vec<Vec<usize>>,
  total_game_results: Vec<i32>,
  current_game_moves: Vec<Vec<usize>>,
}

impl GDEngine {
    pub fn new(rounds_per_game: u8, is_first_player: bool) -> GDEngine {
        GDEngine {
            rounds: rounds_per_game,
            exploration: e,
            moves: vec![],
            positions: vec![],
            scores: empty_scores,
            first_player: is_first_player,
        }
    }
}

impl Engine for GDEngine {
    fn get_id(&self) -> String {
        "gd engine".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      let board_position: [i8; 36] = board.get_board_position();
      let normalized_move_probs: [f32;36] = nn.predict(board_position);
      let proposed_move = random_select(normalized_move_probs);
      if !board.get_possible_moves().contains(proposed_move) {
        nn.penalize_illegal_move(normalized_move);
        propsed_move = // random select legal move
      }
      proposed_move
    }
    
    fn reset_board(&mut self) {
        self.current_game_moves = vec![];
    }

    fn finish_round(&mut self, result: i32) {
      self.total_game_moves.push(current_game_moves);
      self.total_game_results.push(result);
      self.games_played += 1;
    }

}
