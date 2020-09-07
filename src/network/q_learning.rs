use crate::env::env_trait::Environment;
use rand::Rng;
use rand::ThreadRng;
use std::collections::HashMap;
use ndarray::Array1;

#[allow(dead_code)]
pub struct Qlearning {
    exploration: f32,
    learning_rate: f32,
    discount_factor: f32,
    scores: HashMap<(String, usize), f32>, // (State,Action), reward
    last_state: String,
    last_action: usize, 
    rng: ThreadRng,
}

// based on Q-learning using a HashMap as table
//
impl Qlearning {
    pub fn new(exploration: f32) -> Self {
        let learning_rate = 0.1f32;
        let discount_factor = 0.95f32;
        Qlearning {
            exploration,
            learning_rate,
            discount_factor,
            last_action: 42 as usize,
            last_state: "".to_string(),
            scores: HashMap::new(),
            rng: rand::thread_rng(),
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

impl Qlearning {


    pub fn reset_board(&mut self) {}

    // update "table" based on last action and their result
    pub fn finish_round(&mut self, result: i32) {
        // -1 for loss, 0 for draw, 1 for win
        self.update_map(50. * (result as f32), 0.); // 50*res, since the final result is what matters most
    }

    pub fn get_move(&mut self, board: &impl Environment) -> usize {
        //get reward for previous move

        let (board_arr, _actions, reward) = board.step();
        let legal_actions = board.get_legal_actions();
        let (best_move, max_future_q) = self.get_best_move(legal_actions.clone());

        // update HashMap-entry for last (state,action) based on the received reward
        self.update_map(reward, max_future_q);

        self.last_action = best_move;
        self.last_state = board_arr.fold("".to_string(), |acc, x| acc + &x.to_string());

        if self.exploration > rand::thread_rng().gen() {
            self.last_action = self.get_random_move(legal_actions);
        }

        self.last_action
    }

    fn get_random_move(&mut self, actions: Array1<usize>) -> usize {
        let position = self.rng.gen_range(0, actions.len()) as usize;
        return actions[position];
    }

    fn get_best_move(&mut self, actions: Array1<usize>) -> (usize, f32) {
        //42 is illegal board position, would result in error
        let mut best_pair: (usize, f32) = (42, f32::MIN);

        //println!("#possible actions: {}", actions.len());
        for &move_candidate in actions.iter() {
            let score = self
                .scores
                .entry((self.last_state.clone(), move_candidate))
                .or_insert(self.rng.gen_range(-0.5, 0.5));
            if *score > best_pair.1 {
                best_pair = (move_candidate, *score);
            }
        }
        //println!("{:?}", best_pair);
        best_pair
    }

    fn update_map(&mut self, reward: f32, max_future_q: f32) {
        let score = self
            .scores
            .entry((self.last_state.clone(), self.last_action.clone()))
            .or_insert(self.rng.gen_range(-0.5, 0.5));
        *score = (1. - self.learning_rate) * (*score)
            + self.learning_rate * (reward + self.discount_factor * max_future_q);
    }
}
