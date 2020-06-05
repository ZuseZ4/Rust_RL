use rand::Rng;
use std::collections::HashMap;

use crate::board::board_trait::BoardInfo;
use crate::engine::engine_trait::Engine;

#[allow(dead_code)]
pub struct AIEngine {
    rounds: u8,
    exploration: f32,
    scores: HashMap<String, i32>,
    moves: Vec<usize>,
    positions: Vec<String>,
    first_player: bool,
}

impl AIEngine {
    pub fn new(rounds_per_game: u8, is_first_player: bool, e: f32) -> Self {
        let empty_scores = HashMap::new();
        AIEngine {
            rounds: rounds_per_game,
            exploration: e,
            moves: vec![],
            positions: vec![],
            scores: empty_scores,
            first_player: is_first_player,
        }
    }
}

impl Engine for AIEngine {
    fn get_id(&self) -> String {
        "ai engine".to_string()
    }

    fn reset_board(&mut self) {
        self.moves = vec![];
        self.positions = vec![];
    }

    fn finish_round(&mut self, result: i32) {
        for position in self.positions.iter() {
            let score = self.scores.entry(position.to_string()).or_insert(0);
            *score += result;
        }
    }

    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        let (board_strings, moves) = board.get_possible_positions();

        // explore randomly a new move or make probably best move?
        if self.exploration > rand::thread_rng().gen() {
            //println!("exploring a new move");
            let position = rand::thread_rng().gen_range(0, moves.len()) as usize;
            let board_string_copy = board_strings[position].clone();
            self.positions.push(board_string_copy);
            return moves[position];
        }

        // play the best move
        //println!("picking best move");

        //42 is illegal board position, would result in error
        let mut best_pair: (usize, i32, &str) = if self.first_player {(42, i32::MIN, "")} else {(42, i32::MAX, "")};
        let mut new_moves = Vec::new();

        for (board_candidate, move_candidate) in board_strings.iter().zip(moves.iter()) {
            match self.scores.get(board_candidate) {
                Some(&score) => {
                    //TODO randomize order. otherwise always takes first good move
                    if (self.first_player && best_pair.1 < score)
                        || (!self.first_player && best_pair.1 > score)
                    {
                        best_pair = (*move_candidate, score, board_candidate);
                    }
                }
                None => {
                    if (self.first_player && best_pair.1 < 0)
                        || (!self.first_player && best_pair.1 > 0)
                    {
                        new_moves.push((*move_candidate, board_candidate));
                    }
                }
            }
        }

        if (self.first_player && best_pair.1 < 0) || (!self.first_player && best_pair.1 > 0) {
            //if possible better play a unknown move than a bad move
            if !new_moves.is_empty() {
                let move_number = rand::thread_rng().gen_range(0, new_moves.len()) as usize;
                best_pair = (new_moves[move_number].0, 0, new_moves[move_number].1);
            }
        }
        self.positions.push(best_pair.2.to_string());
        best_pair.0
    }
}
