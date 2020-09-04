use crate::network::q_learning::Qlearning;

use crate::board::board_trait::BoardInfo;
use crate::agent::agent_trait::Agent;

#[allow(dead_code)]
pub struct QLAgent {
    qlearning: Qlearning,
    rounds: u8,
    first_player: bool,
}

// based on Q-learning using a HashMap as table
// 
impl QLAgent {
    pub fn new(rounds: u8, first_player: bool, exploration: f32) -> Self {
        QLAgent {
            qlearning: Qlearning::new(exploration),
            rounds,
            first_player,
        }
    }
}

impl Agent for QLAgent {
    fn get_id(&self) -> String {
        "qlearning agent".to_string()
    }

    fn finish_round(&mut self, mut result: i32) { // -1 for loss, 0 for draw, 1 for win
      if !self.first_player {
        result *= -1;
      }
      self.qlearning.finish_round(result);
    }
  

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      self.qlearning.get_move(board)
    }

  fn get_exploration_rate(&self) -> f32 {
    return self.qlearning.get_exploration_rate();
  }
  
  fn set_exploration_rate(&mut self, e: f32) -> Result<(),String>{
    if e < 0. || e > 1. {
      return Err("exploration rate must be in [0,1]!".to_string());
    }
    self.qlearning.set_exploration_rate(e)?;
    Ok(())
  }

}
