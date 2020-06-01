use std::io;
//use crate::board::smart_board::Board;
use crate::board::board_trait::BoardInfo;
use crate::engine::engine_trait::Engine;

#[allow(dead_code)]
pub struct HumanPlayer {
    rounds: u8,
}

impl HumanPlayer {
    pub fn new(rounds_per_game: u8, _is_first_player: bool) -> HumanPlayer {
        HumanPlayer {rounds: rounds_per_game,}
    }
}

impl Engine for HumanPlayer {
    fn get_id(&self) -> String {
        "human player".to_string()
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
      board.print_board();
      let moves = board.get_possible_moves();
      let mut next_move = String::new();
  
      loop {
        println!("please insert the next move.");
        io::stdin()
            .read_line(&mut next_move)
            .expect("Failed to read number of rounds");
        let next_move: usize = next_move.trim().parse().expect("please type a number");
        if moves.contains(&next_move) {return next_move;} 
      }
    }

    fn finish_round(&mut self, _single_res: i32) {}

    fn reset_board(&mut self) {}
}
