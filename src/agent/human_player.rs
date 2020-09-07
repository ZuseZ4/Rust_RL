use crate::agent::agent_trait::Agent;
use crate::env::env_trait::Environment;
use std::io;

#[allow(dead_code)]
pub struct HumanPlayer {
    rounds: u8,
}

impl HumanPlayer {
    pub fn new(rounds: u8, _is_first_player: bool) -> Self {
        HumanPlayer { rounds }
    }
}

impl Agent for HumanPlayer {
    fn get_id(&self) -> String {
        "human player".to_string()
    }

    fn get_move(&mut self, board: &impl Environment) -> usize {
        board.print_board();
        let (_, actions, _) = board.step();
        let mut next_action = String::new();

        loop {
            println!("please insert the number of your next action.\n It should be a number between 1 and {}", actions.len());
            io::stdin()
                .read_line(&mut next_action)
                .expect("Failed to read number of rounds");
            let next_action: usize = next_action.trim().parse().expect("please type a number");
            if next_action >= 1 && next_action <= actions.len() && actions[next_action-1] == 1. {
                return next_action;
            }
        }
    }

    fn finish_round(&mut self, _single_res: i32) {}

    fn get_exploration_rate(&self) -> f32 {
        42.
    }

    fn set_exploration_rate(&mut self, _e: f32) -> Result<(), String> {
        Ok(())
    }
}
