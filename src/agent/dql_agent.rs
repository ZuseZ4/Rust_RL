use crate::network::dq_learning::DQlearning;

use crate::board::board_trait::BoardInfo;
use crate::agent::agent_trait::Agent;

#[allow(dead_code)]
pub struct DQLAgent {
    dqlearning: DQlearning,
    rounds: u8,
    first_player: bool,
}

// based on Q-learning using a HashMap as table
// 
impl DQLAgent {
    pub fn new(rounds: u8, first_player: bool, exploration: f32) -> Self {
        DQLAgent {
            dqlearning: DQlearning::new(exploration),
            rounds,
            first_player,
        }
    }
}

impl Agent for DQLAgent {
    fn get_id(&self) -> String {
        "dqlearning agent".to_string()
    }

    fn reset_board(&mut self) {
      self.dqlearning.reset_board();
    }

    fn finish_round(&mut self, mut result: i32) { // -1 for loss, 0 for draw, 1 for win
      if !self.first_player {
        result *= -1;
      }
      self.dqlearning.finish_round(result);
    }
  

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      self.dqlearning.get_move(board)
    }

  fn get_exploration_rate(&self) -> f32 {
    return self.dqlearning.get_exploration_rate();
  }
  
  fn set_exploration_rate(&mut self, e: f32) -> Result<(),String>{
    if e < 0. || e > 1. {
      return Err("exploration rate must be in [0,1]!".to_string());
    }
    self.dqlearning.set_exploration_rate(e)?;
    Ok(())
  }

}
