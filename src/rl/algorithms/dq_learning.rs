use crate::network::nn::NeuralNetwork;
use crate::rl::algorithms::utils;
use ndarray::{Array, Array1, Array2};
use ndarray_stats::QuantileExt;
use rand::{Rng, ThreadRng};
use super::{Observation, ReplayBuffer};

#[allow(dead_code)]
pub struct DQlearning {
    nn: NeuralNetwork,
    counter: usize,
    sum: usize,
    exploration: f32,
    learning_rate: f32,
    discount_factor: f32,
    last_turn: (Array2<f32>, Array1<f32>, Array1<f32>, usize), // (board before last own move, allowed moves, NN output, move choosen from NN)
    replay_buffer: ReplayBuffer,
    rng: ThreadRng,
    epsilon: f32,
}

impl DQlearning {
    pub fn new(exploration: f32, nn: NeuralNetwork) -> Self {
        let learning_rate = 1e-3;
        let discount_factor = 0.95;
        DQlearning {
            sum: 0,
            counter: 0,
            nn,
            exploration,
            learning_rate,
            last_turn: (
                Array::zeros((0, 0)),
                Array::zeros(0),
                Array::zeros(0),
                42,
            ),
            replay_buffer: ReplayBuffer::new(1, 100_000),
            discount_factor,
            rng: rand::thread_rng(),
            epsilon: 1e-8,
        }
    }

    pub fn get_exploration_rate(&self) -> f32 {
        self.exploration
    }

    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if e < 0. || e > 1. {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.exploration = e;
        Ok(())
    }
}

impl DQlearning {
    // update "table" based on last action and their result
    pub fn finish_round(&mut self, result: i32) {
        // -1 for loss, 0 for draw, 1 for win
        self.learn_from_reward(result as f32 * 7.);
    }

    fn select_move(&mut self, prediction: Array1<f32>) -> usize {
        // TODO verify using argmax
        prediction.argmax().unwrap()
    }

    fn learn_from_reward(&mut self, reward: f32) {
        let input = self.last_turn.0.clone();
        let mut target = self.last_turn.2.clone() * self.last_turn.1.clone(); // last_turn.1 (allowed moves) works as a filter to penalize illegal moves simultaneously
        let mut new_target_val = target[self.last_turn.3] + reward;
        target[self.last_turn.3] += reward; // adjust target vector to adapt to outcome of last move
        if new_target_val > 1. {
            new_target_val = 1.;
        }
        if new_target_val < 0. {
            new_target_val = 0.1;
        }
        target[self.last_turn.3] = new_target_val;
        self.nn.train2d(input, target);
    }

    pub fn get_move(
        &mut self,
        board_arr: Array2<f32>,
        action_arr: Array1<bool>,
        reward: f32,
    ) -> usize {
        let actions = action_arr.mapv(|x| (x as i32) as f32);
        
        self.replay_buffer.add_memory(Observation::new(self.last_turn.0.clone(), self.last_turn.3, board_arr.clone(), reward)); //
        self.learn_from_reward(reward);

        let predicted_moves = self.nn.predict2d(board_arr.clone());
        self.count_illegal_moves(predicted_moves.clone(), actions.clone());
        let legal_predicted_moves = predicted_moves.clone() * actions.clone();
        let mut next_move = self.select_move(legal_predicted_moves.clone());

        // shall we explore a random move and ignore prediction?
        if self.exploration > self.rng.gen() {
            next_move = utils::get_random_true_entry(action_arr);
        }

        // bookkeeping
        self.last_turn = (board_arr, actions, predicted_moves, next_move);

        self.last_turn.3
    }

    fn count_illegal_moves(&mut self, predicted_moves: Array1<f32>, allowed_moves: Array1<f32>) {
        let illegal_moves = allowed_moves.mapv(|x| 1. - x);
        let errors = predicted_moves * illegal_moves;
        let errors = errors
            .iter()
            .fold(0, |err, &val| if val > 0.2 { err + 1 } else { err });
        self.sum += errors;
        self.counter += 1;
        let n = 1000;
        if self.counter % n == 0 {
            println!("errors per {} moves: {}", n, self.sum);
            self.sum = 0;
            self.counter = 0;
        }
    }
}
