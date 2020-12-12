use crate::rl::agent::agent_trait::Agent;
use ndarray::{Array1, Array2};
use spaces::Space;
use std::io;

/// An agent which just shows the user the current environment and lets the user decide about each action.
#[derive(Default)]
pub struct HumanPlayer {}

impl HumanPlayer {
    /// No arguments required since the user has to take over the agents decisions.
    pub fn new() -> Self {
        HumanPlayer {}
    }
}

impl<S: Space, A: Space> Agent<S, A> for HumanPlayer {
    fn get_id(&self) -> String {
        "human player".to_string()
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, _: f32) -> usize {
        let (n, m) = (board.shape()[0], board.shape()[1]);
        for i in 0..n {
            for j in 0..m {
                print!("{} ", board[[i, j]]);
            }
            println!();
        }
        loop {
            let mut next_action = String::new();
            println!("please insert the number of your next action.\n It should be a number between 1 and {}", actions.len());
            println!("{}", actions);
            io::stdin()
                .read_line(&mut next_action)
                .expect("Failed to read number of rounds");
            let mut next_action: usize = next_action.trim().parse().expect("please type a number");
            next_action -= 1; //from human to cs indexing.

            // assert choosen move exists and is legal
            if next_action < actions.len() && actions[next_action] {
                // human(non cs) friendly counting
                return next_action;
            } else {
                eprintln!("The selected move was illegal. Please try again.\n");
            }
        }
    }

    fn finish_round(&mut self, _single_res: i32, _final_state: Array2<f32>) {}

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
