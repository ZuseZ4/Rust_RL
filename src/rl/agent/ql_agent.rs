use crate::rl::algorithms::Qlearning;

use crate::rl::agent::Agent;
use crate::rl::env::Environment;

/// An agent working on a classical q-table.
pub struct QLAgent {
    qlearning: Qlearning,
}

// based on Q-learning using a HashMap as table
//
impl QLAgent {
    /// A constructor with an initial exploration rate. 
    pub fn new(exploration: f32) -> Self {
        QLAgent {
            qlearning: Qlearning::new(exploration),
        }
    }
}

impl Agent for QLAgent {
    fn get_id(&self) -> String {
        "qlearning agent".to_string()
    }

    fn finish_round(&mut self, result: i32) {
        // -1 for loss, 0 for draw, 1 for win
        self.qlearning.finish_round(result);
    }

    fn get_move(&mut self, board: &Box<dyn Environment>) -> usize {
        self.qlearning.get_move(board)
    }

    fn get_exploration_rate(&self) -> f32 {
        self.qlearning.get_exploration_rate()
    }

    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if e < 0. || e > 1. {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.qlearning.set_exploration_rate(e)?;
        Ok(())
    }
}
