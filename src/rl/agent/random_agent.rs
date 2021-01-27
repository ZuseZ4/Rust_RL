use crate::rl::agent::agent_trait::Agent;
use crate::rl::algorithms::utils;
use ndarray::{Array1, Array2};

/// An agent who acts randomly.
///
/// All input is ignored except of the vector of possible actions.
/// All allowed actions are considered with an equal probability.
#[derive(Default)]
pub struct RandomAgent {}

impl RandomAgent {
    /// Returns a new instance of a random acting agent.
    pub fn new() -> Self {
        RandomAgent {}
    }
}

impl Agent for RandomAgent {
    fn get_id(&self) -> String {
        "random agent".to_string()
    }

    fn get_move(&mut self, _: Array2<f32>, actions: Array1<bool>, _: f32) -> usize {
        utils::get_random_true_entry(actions)
    }

    fn finish_round(&mut self, _single_res: i8, _final_state: Array2<f32>) {}

    fn get_learning_rate(&self) -> f32 {
        42.
    }

    fn set_learning_rate(&mut self, _e: f32) -> Result<(), String> {
        Ok(())
    }

    fn get_exploration_rate(&self) -> f32 {
        42.
    }

    fn set_exploration_rate(&mut self, _e: f32) -> Result<(), String> {
        Ok(())
    }
}
