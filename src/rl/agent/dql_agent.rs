use crate::network::nn::NeuralNetwork;
use crate::rl::agent::Agent;
use crate::rl::algorithms::DQlearning;
use ndarray::{Array1, Array2};
use spaces::Space;
use spaces::discrete::NonNegativeIntegers;


pub type State<S>  = <S as Space>::Value;
pub type Action<S> = <S as Space>::Value;

/// An agent using Deep-Q-Learning, based on a small neural network.
pub struct DQLAgent {
    dqlearning: DQlearning,
}

// based on Q-learning using a HashMap as table
//
impl DQLAgent {
    /// A constructor including an initial exploration rate.
    pub fn new(exploration: f32, batch_size: usize, nn: NeuralNetwork) -> Self {
        DQLAgent {
            dqlearning: DQlearning::new(exploration, batch_size, nn),
        }
    }
}

impl<S: Space, A: Space> Agent<S, A> for DQLAgent {
    fn get_id(&self) -> String {
        "dqlearning agent".to_string()
    }

    fn finish_round(&mut self, result: i32, final_state: Array2<f32>) {
        // -1 for loss, 0 for draw, 1 for win
        self.dqlearning.finish_round(result, final_state);
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, reward: f32) -> &Action<A> {
        let res = self.dqlearning.get_move(board, actions, reward);
        res
    }

    fn get_learning_rate(&self) -> f32 {
        self.dqlearning.get_learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        self.dqlearning.set_learning_rate(lr)
    }

    fn get_exploration_rate(&self) -> f32 {
        self.dqlearning.get_exploration_rate()
    }

    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if e < 0. || e > 1. {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.dqlearning.set_exploration_rate(e)?;
        Ok(())
    }
}
