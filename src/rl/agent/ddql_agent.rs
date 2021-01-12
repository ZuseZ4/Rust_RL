use crate::network::nn::NeuralNetwork;
use crate::rl::agent::Agent;
use crate::rl::algorithms::DDQlearning;
use ndarray::{Array1, Array2};

/// An agent using Deep-Q-Learning, based on a small neural network.
pub struct DDQLAgent {
    ddqlearning: DDQlearning,
}

// based on Q-learning using a HashMap as table
//
impl DDQLAgent {
    /// A constructor including an initial exploration rate.
    pub fn new(exploration: f32, batch_size: usize, nn: NeuralNetwork) -> Self {
        DDQLAgent {
            ddqlearning: DDQlearning::new(exploration, batch_size, nn),
        }
    }
}

impl Agent for DDQLAgent {
    fn get_id(&self) -> String {
        "ddqlearning agent".to_string()
    }

    fn finish_round(&mut self, reward: f32, final_state: Array2<f32>) {
        self.ddqlearning.finish_round(reward, final_state);
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, reward: f32) -> usize {
        self.ddqlearning.get_move(board, actions, reward)
    }

    fn get_learning_rate(&self) -> f32 {
        self.ddqlearning.get_learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        self.ddqlearning.set_learning_rate(lr)
    }

    fn get_exploration_rate(&self) -> f32 {
        self.ddqlearning.get_exploration_rate()
    }

    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if e < 0. || e > 1. {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.ddqlearning.set_exploration_rate(e)?;
        Ok(())
    }
}
