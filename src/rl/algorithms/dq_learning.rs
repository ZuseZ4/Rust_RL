use super::{Observation, ReplayBuffer};
use crate::network::nn::NeuralNetwork;
use crate::rl::algorithms::utils;
use ndarray::{par_azip, Array1, Array2, Array4};
use ndarray_stats::QuantileExt;
use rand::rngs::ThreadRng;
use rand::Rng;

#[allow(dead_code)]
pub struct DQlearning {
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

impl DQlearning {
    pub fn new(exploration: f32, batch_size: usize, mut nn: NeuralNetwork) -> Self {
        if nn.get_batch_size() % batch_size != 0 {
            eprintln!(
                "not implemented yet, unsure how to store intermediate vals before weight updates"
            );
            unimplemented!();
        }
        nn.set_batch_size(batch_size); // TODO not working yet, see nn.rs
        let discount_factor = 0.95;
        DQlearning {
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
            replay_buffer: ReplayBuffer::new(batch_size, 100),
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

impl DQlearning {
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

    fn learn(&mut self) {
        if !self.replay_buffer.is_full() {
            return;
        }
        let (s0_vec, actions, s1_vec, mut rewards) = self.replay_buffer.get_memories_SoA();
        let s0_arr = vec_to_arr(s0_vec);
        let s1_arr = vec_to_arr(s1_vec);
        let targets: Array2<f32> = self.nn.predict_batch(s0_arr.clone().into_dyn());
        let future_move_rewards: Array2<f32> = self.nn.predict_batch(s1_arr.into_dyn());
        let max_future_reward: Array1<f32> = get_max_rewards(future_move_rewards);
        rewards += &(self.discount_factor * max_future_reward);
        //update_targets(&mut targets, actions, rewards);
        let targets = update_targets(targets, actions, rewards);
        self.nn.train(s0_arr.into_dyn(), targets.into_dyn());
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

        let board_with_channels = board_arr
            .clone()
            .into_shape((1, board_arr.shape()[0], board_arr.shape()[1]))
            .unwrap();
        let predicted_moves = self.nn.predict3d(board_with_channels);
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

fn update_targets(
    mut targets: Array2<f32>,
    actions: Array1<usize>,
    rewards: Array1<f32>,
) -> Array2<f32> {
    let clamped_rewards = clamp(rewards);

    par_azip!((mut target in targets.outer_iter_mut(), action in &actions, reward in &clamped_rewards) {
        target[*action] = *reward;
    });
    targets
}

fn clamp(new_target_val: Array1<f32>) -> Array1<f32> {
    new_target_val.mapv(|x| {
        if x > 1. {
            x
        } else {
            if x < 0. {
                0.1
            } else {
                x
            }
        }
    })
}

fn vec_to_arr(input: Vec<Array2<f32>>) -> Array4<f32> {
    let (bs, nrows, ncols) = (input.len(), input[0].nrows(), input[0].ncols());
    let mut res = Array4::zeros((bs, 1, nrows, ncols));
    par_azip!((mut out_entry in res.outer_iter_mut(), in_entry in &input) {
      out_entry.assign(in_entry);
    });
    res.into_shape((bs, 1, nrows, ncols)).unwrap()
}

fn get_max_rewards(input: Array2<f32>) -> Array1<f32> {
    let mut res = Array1::zeros(input.nrows());

    par_azip!((mut out_entry in res.outer_iter_mut(), in_entry in input.outer_iter()) {
      out_entry.fill(*in_entry.max().unwrap());
    });
    res
}
