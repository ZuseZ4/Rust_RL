use crate::agent::agent_trait::Agent;
use crate::env::Environment;
use std::io;

#[derive(Default)]
pub struct HumanPlayer {}

impl HumanPlayer {
    pub fn new() -> Self {
        HumanPlayer {}
    }
}

impl Agent for HumanPlayer {
    fn get_id(&self) -> String {
        "human player".to_string()
    }

    fn get_move(&mut self, board: &Box<dyn Environment>) -> usize {
        board.render();
        //let (_, actions, _) = board.step();
        let actions = board.get_legal_actions();
        let mut next_action = String::new();

        loop {
            println!("please insert the number of your next action.\n It should be a number between 1 and {}", actions.len());
            println!("{}", actions);
            io::stdin()
                .read_line(&mut next_action)
                .expect("Failed to read number of rounds");
            let next_action: usize = next_action.trim().parse().expect("please type a number");
            if next_action >= 1 && next_action <= actions.len() {
                // human(non cs) friendly counting
                return actions[next_action - 1];
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
