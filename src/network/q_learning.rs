use rand::Rng;
use std::collections::HashMap;
use crate::board::board_trait::BoardInfo;

#[allow(dead_code)]
pub struct Qlearning {
    exploration: f32,
    scores: HashMap<String, [f32;4]>, // [#wins, #draws, #loss, avg_res]
    actions: Vec<usize>,
    states: Vec<String>,
}

// based on Q-learning using a HashMap as table
// 
impl Qlearning {
    pub fn new(exploration: f32) -> Self {
        Qlearning {
            exploration,
            actions: vec![],
            states: vec![],
            scores: HashMap::new(),
        }
    }
  
    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(),String>{
      if e < 0. || e > 1. {
        return Err("exploration rate must be in [0,1]!".to_string());
      }
      self.exploration = e;
      Ok(())
    }
}


impl Qlearning {

    pub fn reset_board(&mut self) {
      self.actions = vec![];
      self.states = vec![];
    }

    // update "table" based on last actions and their result
    pub fn finish_round(&mut self, result: i32) { // -1 for loss, 0 for draw, 1 for win
        //TODO assert that result is in [-1,0,1] or find a better solution
        for position in self.states.iter() {
            let score = self.scores.entry(position.to_string()).or_insert([0.,0.,0.,0.]);
            score[(result+1) as usize] += 1.; // store the outcome of the game where we visited this position
            score[3] = (-score[0] + score[2]) / (score[0]+score[1]+score[2]); // update total value
        }
    }
  

    pub fn get_move(&mut self, board: &impl BoardInfo) -> usize { //get reward for previous move
        let (board_strings, actions) = board.get_possible_positions(); 

        // explore randomly a new move or make probably best move?
        if self.exploration > rand::thread_rng().gen() {
            let position = rand::thread_rng().gen_range(0, actions.len()) as usize;
            self.states.push(board_strings[position].to_string());
            return actions[position];
        }

        // play the hopefully best move
        //println!("picking best move");

        //42 is illegal board position, would result in error
        let mut best_pair: (usize, f32, &str) = (42, f32::MIN, "");
        let mut new_actions = Vec::new();

        //println!("#possible actions: {}", actions.len());
        for (board_candidate, &move_candidate) in board_strings.iter().zip(actions.iter()) {
            let new_score;
            let known_move: bool;
            match self.scores.get(board_candidate) {
              Some(&score) => {
                new_score = score[3];
                known_move = true;
              }
              None => {
                new_score = 0.; // it's an unknown move, lets just guess it's an average move
                known_move = false;
                new_actions.push((move_candidate, board_candidate));
              }
            }

            if (new_score > best_pair.1) && known_move {
              best_pair = (move_candidate, new_score, board_candidate);
            } 
        }

        //println!("prediction based on known actions: {}", best_pair.1);
        if (0. > best_pair.1) && !new_actions.is_empty() { 
            // if possible select randomly an unknown move rather than a bad one
            let move_number = rand::thread_rng().gen_range(0, new_actions.len()) as usize;
            best_pair = (new_actions[move_number].0, 0., new_actions[move_number].1);
        }
        self.states.push(best_pair.2.to_string());
        best_pair.0
    }
  

}
