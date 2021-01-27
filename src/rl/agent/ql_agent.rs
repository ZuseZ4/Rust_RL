use super::results::RunningResults;
use crate::rl::algorithms::Qlearning;
use ndarray::{Array1, Array2};

use crate::rl::agent::Agent;

/// An agent working on a classical q-table.
pub struct QLAgent {
    qlearning: Qlearning,
    results: RunningResults,
}

// based on Q-learning using a HashMap as table
//
impl QLAgent {
    /// A constructor with an initial exploration rate.
    pub fn new(exploration: f32, learning: f32, action_space_length: usize) -> Self {
        QLAgent {
            qlearning: Qlearning::new(exploration, learning, action_space_length),
            results: RunningResults::new(100, true),
        }
    }
}

impl Agent for QLAgent {
    fn get_id(&self) -> String {
        "qlearning agent".to_string()
    }

    fn finish_round(&mut self, reward: i8, final_state: Array2<f32>) {
        self.results.add(reward.into());
        self.qlearning.finish_round(reward.into(), final_state);
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, reward: f32) -> usize {
        self.qlearning.get_move(board, actions, reward)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&lr) {
            return Err("learning rate must be in [0,1]!".to_string());
        }
        self.qlearning.set_learning_rate(lr)?;
        Ok(())
    }

    fn get_learning_rate(&self) -> f32 {
        self.qlearning.get_learning_rate()
    }

    fn get_exploration_rate(&self) -> f32 {
        self.qlearning.get_exploration_rate()
    }

    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&e) {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.qlearning.set_exploration_rate(e)?;
        Ok(())
    }
}
