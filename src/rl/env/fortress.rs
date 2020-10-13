use crate::rl::env::env_trait::Environment;
use fnv::FnvHashSet;
use ndarray::{Array, Array1, Array2};
use std::collections::HashSet;

/// A struct containing all relevant information to store the current position of a single fortress game.
pub struct Board {
    field: [i8; 36],
    flags: [i8; 36],
    neighbours: [Vec<usize>; 36],
    first_player_turn: bool,
    rounds: u32,
    total_rounds: u8,
    //first_player_moves: Vec<usize>, //TODO HashSet is slower than recalculating possible moves each time. remove it?
    //second_player_moves: Vec<usize>,
    first_player_moves: HashSet<usize, std::hash::BuildHasherDefault<fnv::FnvHasher>>,
    second_player_moves: HashSet<usize, std::hash::BuildHasherDefault<fnv::FnvHasher>>,
}

impl Environment for Board {
    // return values:
    // 6x6 field with current board
    // 6x6 field with 1's for possible moves, 0's for impossible moves
    fn step(&self) -> (Array2<f32>, Array1<f32>, f32) {
        // storing current position into ndarray
        let position = Array2::from_shape_vec((6, 6), self.field.to_vec()).unwrap();
        let position = position.mapv(|x| x as f32);
        let position = position.mapv(|x| (x + 3.) / 6.); // scale to [0,1]

        // collecting allowed moves
        let mut moves = Array::zeros(36); // start as a ndarray of length 36
        let possible_actions = self.get_possible_moves();
        for action in possible_actions {
            moves[action] = 1.;
        }
        //let moves = moves.into_shape((6,6)).unwrap(); // transform into 6x6
        let moves: Array1<f32> = moves.mapv(|x| x as f32); // transform from usize to f32

        // get rewards
        let reward = self.get_reward();

        (position, moves, reward as f32)
    }

    fn get_legal_actions(&self) -> Array1<usize> {
        if self.first_player_turn {
            self.first_player_moves.iter().copied().collect()
        } else {
            self.second_player_moves.iter().copied().collect()
        }
    }

    fn reset(&mut self) {
        *self = Board::new(self.total_rounds);
    }

    fn done(&self) -> bool {
        // just played sufficient moves
        if self.rounds == self.total_rounds as u32 * 2u32 {
            return true;
        }
        // no moves possible so finish game
        if (self.first_player_moves.is_empty() && self.first_player_turn)
            || (self.second_player_moves.is_empty() && !self.first_player_turn)
        {
            return true;
        }
        false
    }

    fn render(&self) {
        let mut fst;
        let mut snd;
        let mut val;
        for row_num in 0..6 {
            println!();
            let mut line = String::from("");
            for x in 0..6 {
                fst = self.field[6 * row_num + x];
                snd = self.flags[6 * row_num + x];
                val = 10 * fst + snd;
                let update = format!(" {:^3}", &val.to_string());
                line.push_str(&update);
            }
            println!("{} ", line);
        }
        println!();
    }

    fn take_action(&mut self, pos: usize) -> bool {
        let player_val = if self.first_player_turn { 1 } else { -1 };

        // check that field is not controlled by enemy, no enemy building on field, no own building on max lv (3) already exists
        if (self.first_player_turn && self.first_player_moves.contains(&pos))
            || (!self.first_player_turn && self.second_player_moves.contains(&pos))
        {
            self.store_update(pos, player_val);
            self.update_neighbours(pos, player_val);
            self.first_player_turn = !self.first_player_turn;
            self.rounds += 1;
            return true;
        }
        self.render();
        let current_player = if self.first_player_turn { 1 } else { 2 };
        eprintln!("WARNING ILLEGAL MOVE {} BY PLAYER {}", pos, current_player);
        false // move wasn't allowed, do nothing
    }

    fn eval(&self) -> Vec<i8> {
        let controlled_fields = self.controlled_fields();
        let mut diff = controlled_fields.0 as i8 - controlled_fields.1 as i8; // diff \in [-36,36]
        if diff != 0 {
            diff = diff / diff.abs();
        }
        match diff {
            -1 => vec![-1, 1],
            0 => vec![0, 0], //draw
            1 => vec![1, -1],
            _ => panic!("false implementation of eval in fortress.rs!"),
        }
    }
}

impl Board {
    /// A simple constructor which just takes the amount of moves from each player during a single game.
    ///
    /// After the given amount of rounds the player which controlls the majority of fields wins a single game.
    pub fn new(total_rounds: u8) -> Self {
        let neighbours_list = [
            vec![1, 6],
            vec![0, 2, 7],
            vec![1, 3, 8],
            vec![2, 4, 9],
            vec![3, 5, 10],
            vec![4, 11],
            vec![7, 0, 12],
            vec![6, 8, 1, 13],
            vec![7, 9, 2, 14],
            vec![8, 10, 3, 15],
            vec![9, 11, 4, 16],
            vec![10, 5, 17],
            vec![13, 6, 18],
            vec![12, 14, 7, 19],
            vec![13, 15, 8, 20],
            vec![14, 16, 9, 21],
            vec![15, 17, 10, 22],
            vec![16, 11, 23],
            vec![19, 12, 24],
            vec![18, 20, 13, 25],
            vec![19, 21, 14, 26],
            vec![20, 22, 15, 27],
            vec![21, 23, 16, 28],
            vec![22, 17, 29],
            vec![25, 18, 30],
            vec![24, 26, 19, 31],
            vec![25, 27, 20, 32],
            vec![26, 28, 21, 33],
            vec![27, 29, 22, 34],
            vec![28, 23, 35],
            vec![31, 24],
            vec![30, 32, 25],
            vec![31, 33, 26],
            vec![32, 34, 27],
            vec![33, 35, 28],
            vec![34, 29],
        ];
        let mut first_player_hashset = FnvHashSet::default();
        let mut second_player_hashset = FnvHashSet::default();
        for i in 0..36 {
            first_player_hashset.insert(i);
            second_player_hashset.insert(i);
        }
        Board {
            field: [0; 36],
            flags: [0; 36],
            neighbours: neighbours_list,
            first_player_turn: true,
            rounds: 0,
            total_rounds,
            first_player_moves: first_player_hashset,
            second_player_moves: second_player_hashset,
        }
    }

    fn update_neighbours(&mut self, pos: usize, update_val: i8) {
        let neighbours: Vec<usize> = self.neighbours[pos].clone().into_iter().collect();
        for neighbour_pos in neighbours {
            self.flags[neighbour_pos] += update_val;
            if self.field[neighbour_pos] * self.flags[neighbour_pos] < 0 {
                //enemy neighbour building outnumbered, destroy it
                let val = -self.field[neighbour_pos];
                self.field[neighbour_pos] = 0;
                self.flags[neighbour_pos] += val;
                self.update_neighbours(neighbour_pos, val);
            }
        }
    }

    fn store_update(&mut self, pos: usize, player_val: i8) {
        self.field[pos] += player_val;
        self.flags[pos] += player_val;
        if self.field[pos].abs() == 3 {
            // buildings on lv. 3 can't be upgraded
            self.second_player_moves.remove(&pos);
            self.first_player_moves.remove(&pos);
            return;
        }
        if player_val == 1 {
            self.second_player_moves.remove(&pos);
        } else {
            self.first_player_moves.remove(&pos);
        }
    }
    fn get_possible_moves(&self) -> Vec<usize> {
        if self.first_player_turn {
            self.first_player_moves.iter().copied().collect()
        } else {
            self.second_player_moves.iter().copied().collect()
        }
    }

    fn get_reward(&self) -> i32 {
        let controlled_fields = self.controlled_fields();
        let mut reward = (controlled_fields.0 as i32) - (controlled_fields.1 as i32);
        if !self.first_player_turn {
            reward *= -1;
        }
        reward
    }

    fn controlled_fields(&self) -> (u8, u8) {
        let mut fields_one = 0;
        let mut fields_two = 0;
        for (&building_lv, &flags) in self.field.iter().zip(self.flags.iter()) {
            if building_lv > 0 || (building_lv == 0 && flags > 0) {
                fields_one += 1;
            } else if building_lv == 0 && flags == 0 {
                continue;
            } else {
                fields_two += 1;
            }
        }
        (fields_one, fields_two)
    }

    /// A getter for the amount of action each player is allowed to take before the game ends.
    pub fn get_total_rounds(&self) -> u8 {
        self.total_rounds
    }
}
