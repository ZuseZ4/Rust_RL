use crate::rl::env::env_trait::Environment;
use ndarray::{Array, Array1, Array2};

#[allow(clippy::unusual_byte_groupings)]
static BITMASKS: [&[u16]; 9] = [
    &[0b_111, 0b_100_100_100, 0b_100_010_001],
    &[0b_111, 0b_010_010_010],
    &[0b_111, 0b_001_010_100, 0b_001_001_001],
    &[0b_111_000, 0b_100_100_100],
    &[0b_111_000, 0b_100_010_001, 0b_010_010_010, 0b_001_010_100],
    &[0b_111_000, 0b_100_100_100],
    &[0b_111_000_000, 0b_001_001_001, 0b_001_010_100],
    &[0b_111_000_000, 0b_010_010_010],
    &[0b_111_000_000, 0b_100_010_001, 0b_001_001_001],
];

/// A struct containing all relevant information to store the current position of a single fortress game.
pub struct TicTacToe {
    player1: u16,
    player2: u16,
    first_player_turn: bool,
    rounds: usize,
    total_rounds: usize,
    state: GameState,
}

#[derive(Debug, Clone, PartialEq)]
enum GameState {
    Running,
    Draw,
    Player1won,
    Player2won,
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for TicTacToe {
    fn step(&self) -> (Array2<f32>, Array1<bool>, f32, bool) {
        // storing current position into ndarray
        let position = board_as_arr(self.player1, self.player2)
            .into_shape((3, 3))
            .unwrap();
        let position = position.mapv(|x| x as f32);

        // collecting allowed moves
        let moves = get_legal_actions(self.player1, self.player2);

        // get rewards
        let agent_num = if self.first_player_turn { 0 } else { 1 };
        let reward = get_reward(&self.state, agent_num);

        let done = self.done();

        (position, moves, reward, done)
    }

    fn reset(&mut self) {
        *self = TicTacToe::new();
    }

    fn render(&self) {
        let res = board_as_arr(self.player1, self.player2);
        for i in 0..3 {
            println!("{} {} {}", res[3 * i + 0], res[3 * i + 1], res[3 * i + 2]);
        }
    }

    fn eval(&mut self) -> Vec<i8> {
        match self.state {
            GameState::Player1won => vec![1, -1],
            GameState::Player2won => vec![-1, 1],
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

        self.state = check_result(self.first_player_turn, self.player1, self.player2, pos);

        self.first_player_turn ^= true; // next player

        true
    }
}

impl TicTacToe {
    /// A simple constructor which just takes the amount of moves from each player during a single game.
    ///
    /// If one player achieves it to put 3 pieces in a row he wins, otherwise the game ends as a draw.
    pub fn new() -> Self {
        TicTacToe {
            player1: 0u16,
            player2: 0u16,
            first_player_turn: true,
            rounds: 0,
            total_rounds: 9,
            state: GameState::Running,
        }
    }

    fn done(&self) -> bool {
        !matches!(self.state, GameState::Running)
    }

    /// A getter for the amount of action each player is allowed to take before the game ends.
    pub fn get_total_rounds(&self) -> usize {
        self.total_rounds
    }
}

fn board_as_arr(player1: u16, player2: u16) -> Array1<i32> {
    let p1_at = |x: u16| -> i32 {
        if (player1 & 1 << x) != 0 {
            1
        } else {
            0
        }
    };
    let p2_at = |x: u16| -> i32 {
        if (player2 & 1 << x) != 0 {
            1
        } else {
            0
        }
    };
    let val_at = |x: u16| -> i32 { p1_at(x) - p2_at(x) };
    let mut res = Array::zeros(9);
    for i in 0..9u16 {
        res[i as usize] = val_at(i);
    }
    res
}

fn get_legal_actions(player1: u16, player2: u16) -> Array1<bool> {
    let bit_res = 0b_111_111_111 & (!(player1 | player2));
    let mut res = Array::from_elem(9, true);
    for i in 0..9 {
        res[i] = (bit_res & 1 << i) != 0;
    }
    res
}

// Game fields are:
// 0,1,2,
// 3,4,5,
// 6,7,8
// pos i is encoded at bit 2^i
#[test]
fn test_bitmask() {
    let p1 = GameState::Player1won;
    let p2 = GameState::Player2won;
    let r = GameState::Running;
    let (fr, _mr, _lr) = (1 + 2 + 4, 8 + 16 + 32, 64 + 128 + 256); //rows
    let (_fc, mc, _lc) = (1 + 8 + 64, 2 + 16 + 128, 4 + 32 + 256); //columns
    let (tlbr, trbl) = (1 + 16 + 256, 4 + 16 + 64); //diagonals
    assert_eq!(p1, check_result(true, fr, 0u16, 0));
    assert_eq!(p1, check_result(true, fr, 0u16, 1));
    assert_eq!(p1, check_result(true, fr, 0u16, 2));
    assert_eq!(r, check_result(true, fr, 0u16, 3));
    assert_eq!(p1, check_result(true, mc, 0u16, 1));
    assert_eq!(p1, check_result(true, mc, 0u16, 4));
    assert_eq!(p1, check_result(true, mc, 0u16, 7));
    assert_eq!(r, check_result(true, mc, 0u16, 5));
    assert_eq!(p1, check_result(true, tlbr, 0u16, 0));
    assert_eq!(p1, check_result(true, tlbr, 0u16, 4));
    assert_eq!(p1, check_result(true, tlbr, 0u16, 8));
    assert_eq!(r, check_result(true, tlbr, 0u16, 1));
    assert_eq!(p2, check_result(false, 0u16, trbl, 2));
    assert_eq!(p2, check_result(false, 0u16, trbl, 4));
    assert_eq!(p2, check_result(false, 0u16, trbl, 6));
    assert_eq!(r, check_result(false, 0u16, trbl, 0));
}

fn check_result(first_player_turn: bool, player1: u16, player2: u16, pos: usize) -> GameState {
    let board = if first_player_turn { player1 } else { player2 };

    for &bm in BITMASKS[pos] {
        assert!(bm < 512);
        if (board & bm) == bm {
            if first_player_turn {
                return GameState::Player1won;
            } else {
                return GameState::Player2won;
            };
        }
    }

    if (player1 | player2) == 511 {
        return GameState::Draw;
    }
    GameState::Running
}

// For a higher complexity we give rewards only when finishing games
fn get_reward(_state: &GameState, _agent_num: usize) -> f32 {
    0.
    /*
    let x = if agent_num == 0 { 1. } else { -1. };
    match state {
        GameState::Draw => 0.4,
        GameState::Player1won => 1. * x,
        GameState::Player2won => -1. * x,
        GameState::Running => 0.,
    }
    */
}
