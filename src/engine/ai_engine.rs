use rand::Rng;
use std::collections::HashMap;

use crate::board::board_trait::BoardInfo;
use crate::engine::engine_trait::Engine;

#[allow(dead_code)]
pub struct AIEngine {
    rounds: u8,
    exploration: f32,
    scores: HashMap<String, [i32;3]>, // [#wins, #draws, #loss]
    moves: Vec<usize>,
    positions: Vec<String>,
    first_player: bool,
}

// based on Q-learning using a HashMap as table
// 
impl AIEngine {
    pub fn new(rounds: u8, first_player: bool, exploration: f32) -> Self {
        AIEngine {
            rounds,
            exploration,
            moves: vec![],
            positions: vec![],
            scores: HashMap::new(),
            first_player,
        }
    }

  fn new_reward_over_thresh(&self, new_val: f32, thresh: f32) -> bool {
    if (self.first_player && new_val > thresh) || (!self.first_player && new_val < thresh) {
      return true;
    }
    false
  }

}

fn get_average_score(scores: [i32;3]) -> f32 {
  let sum: i32 = -1 * scores[0] + 1 * scores[2]; // ignore draws
  let num_games: i32 = scores.iter().sum();
  let expected_score = (sum as f32) / (num_games as f32);
  expected_score
}

impl Engine for AIEngine {
    fn get_id(&self) -> String {
        "ai engine".to_string()
    }

    fn reset_board(&mut self) {
        self.moves = vec![];
        self.positions = vec![];
    }

    fn finish_round(&mut self, result: i32) { // -1 for loss, 0 for draw, 1 for win
        for position in self.positions.iter() {
            let score = self.scores.entry(position.to_string()).or_insert([0,0,0]);
            score[(result+1) as usize] += 1; // store the outcome of the game where we visited this position
        }
    }
  

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        let (board_strings, moves) = board.get_possible_positions();

        // explore randomly a new move or make probably best move?
        if self.exploration > rand::thread_rng().gen() {
            let position = rand::thread_rng().gen_range(0, moves.len()) as usize;
            self.positions.push(board_strings[position].to_string());
            return moves[position];
        }

        // play the hopefully best move
        //println!("picking best move");

        //42 is illegal board position, would result in error
        let mut best_pair: (usize, f32, &str) = if self.first_player {(42, f32::MIN, "")} else {(42, f32::MAX, "")};
        let mut new_moves = Vec::new();

        //println!("#possible moves: {}", moves.len());
        for (board_candidate, &move_candidate) in board_strings.iter().zip(moves.iter()) {
            let new_score;
            let known_move: bool;
            match self.scores.get(board_candidate) {
              Some(&score) => {
                new_score = get_average_score(score);
                known_move = true;
              }
              None => {
                new_score = 0.; // it's an unknown move, lets just guess it's an average move
                known_move = false;
                new_moves.push((move_candidate, board_candidate));
              }
            }

            if self.new_reward_over_thresh(new_score, best_pair.1) && known_move {
              best_pair = (move_candidate, new_score, board_candidate);
            } 
        }

        //println!("prediction based on known moves: {}", best_pair.1);
        if self.new_reward_over_thresh(0., best_pair.1) && !new_moves.is_empty() { 
            // if possible select randomly an unknown move rather than a bad one
            let move_number = rand::thread_rng().gen_range(0, new_moves.len()) as usize;
            best_pair = (new_moves[move_number].0, 0., new_moves[move_number].1);
        }
        self.positions.push(best_pair.2.to_string());
        best_pair.0
    }
  
  fn set_exploration_rate(&mut self, e: f32) -> Result<(),String>{
    if e < 0. || e > 1. {
      return Err("exploration rate must be in [0,1]!".to_string());
    }
    self.exploration = e;
    Ok(())
  }

}

