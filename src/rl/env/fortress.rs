use crate::rl::env::env_trait::Environment;
use ndarray::{Array, Array1, Array2};
use std::cmp::Ordering;

static NEIGHBOURS_LIST: [&[usize]; 6 * 6] = [
    &[1, 6],
    &[0, 2, 7],
    &[1, 3, 8],
    &[2, 4, 9],
    &[3, 5, 10],
    &[4, 11],
    &[7, 0, 12],
    &[6, 8, 1, 13],
    &[7, 9, 2, 14],
    &[8, 10, 3, 15],
    &[9, 11, 4, 16],
    &[10, 5, 17],
    &[13, 6, 18],
    &[12, 14, 7, 19],
    &[13, 15, 8, 20],
    &[14, 16, 9, 21],
    &[15, 17, 10, 22],
    &[16, 11, 23],
    &[19, 12, 24],
    &[18, 20, 13, 25],
    &[19, 21, 14, 26],
    &[20, 22, 15, 27],
    &[21, 23, 16, 28],
    &[22, 17, 29],
    &[25, 18, 30],
    &[24, 26, 19, 31],
    &[25, 27, 20, 32],
    &[26, 28, 21, 33],
    &[27, 29, 22, 34],
    &[28, 23, 35],
    &[31, 24],
    &[30, 32, 25],
    &[31, 33, 26],
    &[32, 34, 27],
    &[33, 35, 28],
    &[34, 29],
];

/// A struct containing all relevant information to store the current position of a single fortress game.
pub struct Fortress {
    field: [i8; 36],
    flags: [i8; 36],
    first_player_turn: bool,
    rounds: usize,
    total_rounds: usize,
    first_player_moves: Array1<bool>,
    second_player_moves: Array1<bool>,
    active: bool,
}

impl Environment for Fortress {
    fn step(&self) -> (Array2<f32>, Array1<bool>, f32, bool) {
        if !self.active {
            eprintln!("Warning, calling step() after done = true!");
        }

        // storing current position into ndarray
        let position = Array2::from_shape_vec((6, 6), self.field.to_vec()).unwrap();
        let position = position.mapv(|x| x as f32);
        let position = position.mapv(|x| (x + 3.) / 6.); // scale to [0,1]

        // collecting allowed moves
        let moves = self.get_legal_actions();

        // get rewards
        let reward = get_reward(self.first_player_turn, self.field, self.flags);

        let done = self.done();
        (position, moves, reward, done)
    }

    fn reset(&mut self) {
        *self = Fortress::new(self.total_rounds);
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
        if (self.first_player_turn && self.first_player_moves[pos])
            || (!self.first_player_turn && self.second_player_moves[pos])
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

    fn eval(&mut self) -> Vec<i8> {
        if !self.done() {
            panic!("Hey, wait till the game is finished!");
        }
        let (p1, p2) = controlled_fields(self.field, self.flags);
        match p1.cmp(&p2) {
            Ordering::Equal => vec![0, 0],
            Ordering::Greater => vec![1, -1],
            Ordering::Less => vec![-1, 1],
        }
    }
}

impl Fortress {
    /// A simple constructor which just takes the amount of moves from each player during a single game.
    ///
    /// After the given amount of rounds the player which controlls the majority of fields wins a single game.
    pub fn new(total_rounds: usize) -> Self {
        Fortress {
            field: [0; 36],
            flags: [0; 36],
            first_player_turn: true,
            rounds: 0,
            total_rounds,
            first_player_moves: Array::from_elem(36, true),
            second_player_moves: Array::from_elem(36, true),
            active: true,
        }
    }

    /// A getter for the amount of action each player is allowed to take before the game ends.
    pub fn get_total_rounds(&self) -> usize {
        self.total_rounds
    }

    fn done(&self) -> bool {
        self.rounds == self.total_rounds
    }

    fn get_legal_actions(&self) -> Array1<bool> {
        if self.first_player_turn {
            self.first_player_moves.clone()
        } else {
            self.second_player_moves.clone()
        }
    }

    fn update_neighbours(&mut self, pos: usize, update_val: i8) {
        let neighbours: Vec<usize> = NEIGHBOURS_LIST[pos].to_vec();
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
            self.second_player_moves[pos] = false;
            self.first_player_moves[pos] = false;
            return;
        }
        if player_val == 1 {
            self.second_player_moves[pos] = false;
        } else {
            self.first_player_moves[pos] = false;
        }
    }
}

fn get_reward(first_player_turn: bool, field: [i8; 36], flags: [i8; 36]) -> f32 {
    let controlled_fields = controlled_fields(field, flags);
    let mut reward = (controlled_fields.0 as i32) - (controlled_fields.1 as i32);
    if !first_player_turn {
        reward *= -1;
    }
    if reward == 0 {
        // add small bonus for achieving at least a draw
        0.5
    } else {
        reward as f32
    }
}

fn controlled_fields(field: [i8; 36], flags: [i8; 36]) -> (u8, u8) {
    let mut fields_one = 0;
    let mut fields_two = 0;
    for (&building_lv, &flags) in field.iter().zip(flags.iter()) {
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
