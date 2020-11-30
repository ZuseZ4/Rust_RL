use super::{Observation, ReplayBuffer};
use crate::network::nn::NeuralNetwork;
use crate::rl::algorithms::utils;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use rand::rngs::ThreadRng;
use rand::Rng;

#[allow(dead_code)]
pub struct DDQlearning {
    nn: NeuralNetwork,
    counter: usize,
    sum: usize,
    exploration: f32,
    discount_factor: f32,
    last_turn: (Array2<f32>, Array1<f32>, Array1<f32>, usize), // (board before last own move, allowed moves, NN output, move choosen from NN)
    replay_buffer: ReplayBuffer<Array2<f32>>,
    rng: ThreadRng,
    epsilon: f32,
}

impl DDQlearning {
    pub fn new(exploration: f32, mut nn: NeuralNetwork) -> Self {
        let bs = 16;
        nn.set_batch_size(bs);
        let discount_factor = 0.95;
        DDQlearning {
            sum: 0,
            counter: 0,
            nn,
            exploration,
            last_turn: (
                Default::default(),
                Default::default(),
                Default::default(),
                42,
            ),
            replay_buffer: ReplayBuffer::new(bs, 100),
            discount_factor,
            rng: rand::thread_rng(),
            epsilon: 1e-8,
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.nn.get_learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if lr < 0. || lr > 1. {
            return Err("learning rate must be in [0,1]!".to_string());
        }
        self.nn.set_learning_rate(lr);
        Ok(())
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

impl DDQlearning {
    // update "table" based on last action and their result
    pub fn finish_round(&mut self, result: i32, s1: Array2<f32>) {
        // result is -1 for loss, 0 for draw, 1 for win
        let reward = result as f32 * 7.;
        self.replay_buffer.add_memory(Observation::new(
            self.last_turn.0.clone(),
            self.last_turn.3,
            s1,
            reward,
        )); //
        self.learn();
    }

    fn select_move(&mut self, prediction: Array1<f32>) -> usize {
        let bestmove = prediction.argmax().unwrap();
        if prediction[bestmove] < 0.001 {
            eprintln!("warning, nn predictions close to zero!");
        }
        bestmove
    }

    fn clamp(&self, mut new_target_val: f32) -> f32 {
        if new_target_val > 1. {
            new_target_val = 1.;
        }
        if new_target_val < 0. {
            new_target_val = 0.1;
        }
        new_target_val
    }

    fn learn(&mut self) {
        if !self.replay_buffer.is_full() {
            return;
        }
        let memories = self.replay_buffer.get_memories();
        for observation in memories {
            let Observation { s0, a, s1, r } = *observation;
            let mut target = self.nn.predict(s0.clone().into_dyn());
            //assert_eq!(len(a), len(target), "error, nn output not equal to amount of possible actions!");
            let future_move_rewards = self.nn.predict(s1.into_dyn());
            let max_future_reward = future_move_rewards.max().unwrap();
            let new_reward = r + self.discount_factor * max_future_reward;
            target[a] = self.clamp(new_reward);
            self.nn.train(s0.into_dyn(), target);
        }
    }

    pub fn get_move(
        &mut self,
        board_arr: Array2<f32>,
        action_arr: Array1<bool>,
        reward: f32,
    ) -> usize {
        let actions = action_arr.mapv(|x| if x { 1. } else { 0. });

        self.replay_buffer.add_memory(Observation::new(
            self.last_turn.0.clone(),
            self.last_turn.3,
            board_arr.clone(),
            reward,
        )); //
        self.learn();

        let predicted_moves = self.nn.predict2d(board_arr.clone());
        self.count_illegal_moves(predicted_moves.clone(), actions.clone());
        let legal_predicted_moves = predicted_moves.clone() * actions.clone();
        let mut next_move = self.select_move(legal_predicted_moves);

        // shall we explore a random move?
        // also select random move if predicted move not allowed (e.g. legal_predicted_moves contains only 0's).
        if (self.exploration > self.rng.gen()) || (!action_arr[next_move]) {
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
