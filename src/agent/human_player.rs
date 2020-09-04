use std::io;
use crate::board::board_trait::BoardInfo;
use crate::agent::agent_trait::Agent;

#[allow(dead_code)]
pub struct HumanPlayer {
    rounds: u8,
}

impl HumanPlayer {
    pub fn new(rounds: u8, _is_first_player: bool) -> Self {
        HumanPlayer {rounds,}
    }
}

impl Agent for HumanPlayer {
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

    fn get_exploration_rate(&self) -> f32 {
      42.
    }

    fn set_exploration_rate(&mut self, _e: f32) -> Result<(),String>{
      Ok(())
    }
}
