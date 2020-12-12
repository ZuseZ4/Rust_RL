use crate::rl::algorithms::Qlearning;
use ndarray::{Array1, Array2};

use crate::rl::agent::Agent;
use spaces::Space;

pub type State<S>  = <S as Space>::Value;
pub type Action<S> = <S as Space>::Value;

/// An agent working on a classical q-table.
pub struct QLAgent {
    qlearning: Qlearning,
}

// based on Q-learning using a HashMap as table
//
impl QLAgent {
    /// A constructor with an initial exploration rate.
    pub fn new(exploration: f32, action_space_length: usize) -> Self {
        QLAgent {
            qlearning: Qlearning::new(exploration, action_space_length),
        }
    }
}

impl<S: Space, A: Space> Agent<S, A> for QLAgent {
    fn get_id(&self) -> String {
        "qlearning agent".to_string()
    }

    fn finish_round(&mut self, result: i32, final_state: Array2<f32>) {
        // -1 for loss, 0 for draw, 1 for win
        self.qlearning.finish_round(result, final_state);
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, reward: f32) -> &Action<A> {
        let res = self.qlearning.get_move(board, actions, reward);
        res
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if lr < 0. || lr > 1. {
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
        if e < 0. || e > 1. {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.qlearning.set_exploration_rate(e)?;
        Ok(())
    }
}
