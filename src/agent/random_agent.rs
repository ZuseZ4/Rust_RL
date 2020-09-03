use rand::Rng;

use crate::board::board_trait::BoardInfo;
use crate::agent::agent_trait::Agent;

#[allow(dead_code)]
pub struct RandomAgent {
    rounds: u8,
}

impl RandomAgent {
    pub fn new(rounds: u8, _is_first_player: bool) -> Self {
        RandomAgent {rounds: rounds,}
    }
}

impl Agent for RandomAgent {
    fn get_id(&self) -> String {
        "random agent".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        let moves = board.get_possible_moves();
        let move_number = rand::thread_rng().gen_range(0, moves.len()) as usize;
        moves[move_number]
    }

    fn finish_round(&mut self, _single_res: i32) {}

    fn reset_board(&mut self) {}

    fn get_exploration_rate(&self) -> f32 {
      42.
    }

    fn set_exploration_rate(&mut self, _e: f32) -> Result<(),String>{
      Ok(())
    }
}
