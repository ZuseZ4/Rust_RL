use super::{Observation, ReplayBuffer};
use crate::rl::algorithms::utils;
use ndarray::{Array1, Array2};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::HashMap;

#[allow(dead_code)]
pub struct Qlearning {
    exploration: f32,
    learning_rate: f32,
    discount_factor: f32,
    scores: HashMap<(String, usize), f32>, // (State,Action), reward
    replay_buffer: ReplayBuffer<String>,
    last_state: String,
    last_action: usize,
    rng: ThreadRng,
    action_space_length: usize,
}

const EPSILON: f32 = 1e-4;

// based on Q-learning using a HashMap as table
//
impl Qlearning {
    pub fn new(exploration: f32, learning_rate: f32, action_space_length: usize) -> Self {
        let bs = 16;
        let discount_factor = 0.95;
        Qlearning {
            exploration,
            learning_rate,
            discount_factor,
            last_action: 42usize,
            last_state: "".to_string(),
            replay_buffer: ReplayBuffer::new(bs, 1000),
            scores: HashMap::new(),
            rng: rand::thread_rng(),
            action_space_length,
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&lr) {
            return Err("learning rate must be in [0,1]!".to_string());
        }
        self.learning_rate = lr;
        Ok(())
    }
    pub fn get_exploration_rate(&self) -> f32 {
        self.exploration
    }

    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&e) {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.exploration = e;
        Ok(())
    }
}

impl Qlearning {
    // update "table" based on last action and their result
    pub fn finish_round(&mut self, mut reward: f32, s1: Array2<f32>) {
        if (reward).abs() < EPSILON {
            reward = 0.5;
        }
        let s1 = s1.fold("".to_string(), |acc, x| acc + &x.to_string());
        self.replay_buffer.add_memory(Observation::new(
            self.last_state.clone(),
            self.last_action,
            s1,
            reward,
            true,
        ));
        self.learn();
    }

    fn max_future_q(&self, s: String) -> f32 {
        let mut max_val = f32::MIN;
        for a in 0..self.action_space_length {
            let key = (s.clone(), a);
            let new_val = if !self.scores.contains_key(&key) {
                0.
            }
            // equals zero-init with RIC (see learn() regarding RIC).
            else {
                *self.scores.get(&key).expect("can't fail")
            };
            if new_val > max_val {
                max_val = new_val;
            }
        }
        max_val
    }

    fn learn(&mut self) {
        if !self.replay_buffer.is_full() {
            return;
        }
        let mut updates: Vec<((String, usize), f32)> = Default::default();
        let memories = self.replay_buffer.get_memories();
        for observation in memories {
            let Observation { s0, a, s1, r, .. } = *observation;
            let key = (s0, a);
            let new_val = if self.scores.contains_key(&key) {
                let val = self.scores.get(&key).expect("can't fail");
                val + self.learning_rate * (r + self.discount_factor * self.max_future_q(s1) - val)
            } else {
                r // use RIC: https://en.wikipedia.org/wiki/Q-learning#Initial_conditions_(Q0)
            };
            updates.push((key, new_val));
        }
        for (key, new_val) in updates {
            self.scores.insert(key, new_val);
        }
    }

    pub fn get_move(
        &mut self,
        board_arr: Array2<f32>,   // TODO work on T
        action_arr: Array1<bool>, // TODO work on V
        reward: f32,
    ) -> usize {
        let board_as_string = board_arr.fold("".to_string(), |acc, x| acc + &x.to_string());
        if f32::abs(reward) > EPSILON || self.rng.gen::<f32>() < 0.2 {
            self.replay_buffer.add_memory(Observation::new(
                self.last_state.clone(),
                self.last_action,
                board_as_string.clone(),
                reward,
                false, // aparently we are not done yet.
            ));
        }
        self.learn();

        self.last_state = board_as_string;

        if self.exploration > rand::thread_rng().gen() {
            self.last_action = utils::get_random_true_entry(action_arr);
        } else {
            self.last_action = match self.get_best_move(action_arr.clone()) {
                Some(action) => action,
                None => utils::get_random_true_entry(action_arr),
            }
        }

        self.last_action
    }

    fn get_best_move(&mut self, actions: Array1<bool>) -> Option<usize> {
        // get all legal actions
        let mut existing_entries: Vec<(usize, f32)> = Vec::new(); // Vec<(action, val)>
        for move_candidate in 0..actions.len() {
            if !actions[move_candidate] {
                continue;
            }

            let score = self.scores.get(&(self.last_state.clone(), move_candidate));
            if let Some(&val) = score {
                existing_entries.push((move_candidate, val))
            }
        }

        // all state,action pairs are unknown, return any
        if existing_entries.is_empty() {
            return None;
        }

        let (mut pos, mut max_val): (usize, f32) = (0, f32::MIN);
        for (i, (_action, new_val)) in existing_entries.iter().enumerate() {
            if *new_val > max_val {
                pos = i;
                max_val = *new_val;
            }
        }

        Some(existing_entries[pos].0)
    }
}
