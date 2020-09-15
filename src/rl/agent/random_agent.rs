use rand::Rng;

use crate::rl::agent::agent_trait::Agent;
use crate::rl::env::Environment;

#[derive(Default)]
pub struct RandomAgent {}

impl RandomAgent {
    pub fn new() -> Self {
        RandomAgent {}
    }
}

impl Agent for RandomAgent {
    fn get_id(&self) -> String {
        "random agent".to_string()
    }

    fn get_move(&mut self, board: &Box<dyn Environment>) -> usize {
        let (_, actions, _) = board.step();
        let num_legal_actions = actions.iter().sum::<f32>() as usize;
        assert!(num_legal_actions > 0, "no legal action available!");
        let mut action_number = rand::thread_rng().gen_range(0, num_legal_actions) as usize;
        // find the n'th legal action
        let mut i = 0;

        loop {
            if i >= actions.len() {
                panic!("Illegal code section in RandomAgent. rand function broken?");
            }
            while actions[i] != 1. {
                i += 1;
            }
            if action_number == 0 {
                //println!("making move {} {} {}", i, num_legal_actions, actions);
                return i;
            } else {
                action_number -= 1;
                i += 1;
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
