use rand::Rng;

use crate::board::board_trait::BoardInfo;
//use crate::board::smart_board::Board;
use crate::engine::engine_trait::Engine;

#[allow(dead_code)]
pub struct RandomEngine {
    rounds: u8,
}

impl RandomEngine {
    pub fn new(rounds_per_game: u8, _is_first_player: bool) -> RandomEngine {
        RandomEngine {rounds: rounds_per_game,}
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
}