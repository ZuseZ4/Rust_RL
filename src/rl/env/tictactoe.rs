use crate::rl::env::env_trait::Environment;
use ndarray::{Array, Array1, Array2};

/// A struct containing all relevant information to store the current position of a single fortress game.
pub struct TicTacToe {
    player1: u16,
    player2: u16,
    bitmasks: [Vec<u16>; 9],
    first_player_turn: bool,
    rounds: usize,
    total_rounds: usize,
    state: GameState,
}
enum GameState {
    Running,
    Draw,
    Player1won,
    Player2won,
}

impl Environment for TicTacToe {
    fn step(&self) -> (Array2<f32>, Array1<bool>, f32, bool) {
        // storing current position into ndarray
        let position = self.board_as_arr().into_shape((3, 3)).unwrap();
        let position = position.mapv(|x| x as f32);

        // collecting allowed moves
        let moves = self.get_legal_actions();

        // get rewards
        let reward = self.get_reward();

        let done = self.done();

        (position, moves, reward as f32, done)
    }

    fn reset(&mut self) {
        *self = TicTacToe::new();
    }

    fn render(&self) {
        let res = self.board_as_arr();
        for i in 0..3 {
            println!("{} {} {}", res[3 * i + 0], res[3 * i + 1], res[3 * i + 2]);
        }
    }

    fn eval(&mut self) -> Vec<i8> {
        match self.state {
            GameState::Player2won => vec![-1, 1],
            GameState::Player1won => vec![1, -1],
            GameState::Draw => vec![0, 0],
            GameState::Running => panic!("Hey, wait till the game is finished!"),
        }
    }

    fn take_action(&mut self, pos: usize) -> bool {
        if pos > 8 {
            return false;
        }
        let bin_pos = 1 << pos;

        if (self.player1 | self.player2) & bin_pos != 0 {
            return false;
        }

        if self.first_player_turn {
            self.player1 |= bin_pos;
        } else {
            self.player2 |= bin_pos;
        }
        self.rounds += 1;

        self.check_result(pos);

        self.first_player_turn ^= true; // next player

        return true;
    }
}

impl TicTacToe {
    /// A simple constructor which just takes the amount of moves from each player during a single game.
    ///
    /// After the given amount of rounds the player which controlls the majority of fields wins a single game.
    pub fn new() -> Self {
        let bitmasks = [
            vec![0b111, 0b1001001, 0b100010001],
            vec![0b111, 0b010010010],
            vec![0b111, 0b0010101, 0b001001001],
            vec![0b000111, 0b100100100],
            vec![0b000111, 0b100010001, 0b010010010, 0b0010101],
            vec![0b000111, 0b001001001],
            vec![0b000000111, 0b100100100, 0b0010101],
            vec![0b000000111, 0b010010010],
            vec![0b000000111, 0b100010001, 0b001001001],
        ];
        TicTacToe {
            player1: 0u16,
            player2: 0u16,
            bitmasks,
            first_player_turn: true,
            rounds: 0,
            total_rounds: 9,
            state: GameState::Running,
        }
    }

    fn board_as_arr(&self) -> Array1<i32> {
        let p1_at = |x: u16| -> i32 {
            if (self.player1 & 2 << x) != 0 {
                1
            } else {
                0
            }
        };
        let p2_at = |x: u16| -> i32 {
            if (self.player2 & 2 << x) != 0 {
                1
            } else {
                0
            }
        };
        let val_at = |x: u16| -> i32 { p1_at(x) - p2_at(x) };
        let mut res = Array::zeros(9);
        for i in 0..9 {
            res[i] = val_at(i as u16);
        }
        res
    }

    fn get_legal_actions(&self) -> Array1<bool> {
        let bit_res = 0b111111111 & (!(self.player1 | self.player2));
        let mut res = Array::from_elem(9, true);
        for i in 0..9 {
            res[i] = (bit_res & 1 << i) != 0;
        }
        res
    }

    fn done(&self) -> bool {
        !matches!(self.state, GameState::Running)
    }

    fn check_result(&mut self, pos: usize) {
        let board = if self.first_player_turn {
            self.player1
        } else {
            self.player2
        };

        for &bm in &self.bitmasks[pos] {
            if (board & bm) == bm {
                self.state = if self.first_player_turn {
                    GameState::Player1won
                } else {
                    GameState::Player2won
                };
                return;
            }
        }

        if (self.player1 | self.player2) == 511 {
            self.state = GameState::Draw;
        }
    }

    fn get_reward(&self) -> f32 {
        let x = if self.first_player_turn { 1. } else { -1. };
        match self.state {
            GameState::Draw => 0.3 * x,
            GameState::Player1won => 1. * x,
            GameState::Player2won => -1. * x,
            GameState::Running => 0. * x,
        }
    }

    /// A getter for the amount of action each player is allowed to take before the game ends.
    pub fn get_total_rounds(&self) -> usize {
        self.total_rounds
    }
}
