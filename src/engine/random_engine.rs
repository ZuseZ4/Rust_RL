use rand::Rng;

use crate::board::board_trait::BoardInfo;
use crate::engine::engine_trait::Engine;

#[allow(dead_code)]
pub struct RandomEngine {
    rounds: u8,
}

impl RandomEngine {
    pub fn new(rounds: u8, _is_first_player: bool) -> Self {
        RandomEngine {rounds: rounds,}
    }
}

impl Engine for RandomEngine {
    fn get_id(&self) -> String {
        "random engine".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        let moves = board.get_possible_moves();
        let move_number = rand::thread_rng().gen_range(0, moves.len()) as usize;
        moves[move_number]
    }

    fn finish_round(&mut self, _single_res: i32) {}

    fn reset_board(&mut self) {}
    
    fn set_exploration_rate(&mut self, _e: f32) -> Result<(),String>{
      Ok(())
    }
}
