use std::collections::HashSet;
use crate::board::board_trait::BoardInfo;
use fnv::FnvHashSet;

pub struct Board {
    field: [i8; 36],
    flags: [i8; 36],
    neighbours: [Vec<usize>; 36],
    first_player_turn: bool,
    total_rounds: usize,
    //first_player_moves: Vec<usize>, //TODO HashSet is slower than recalculating possible moves each time. remove it?
    //second_player_moves: Vec<usize>,
    first_player_moves: HashSet<usize, std::hash::BuildHasherDefault<fnv::FnvHasher>>,
    second_player_moves: HashSet<usize, std::hash::BuildHasherDefault<fnv::FnvHasher>>,
}

impl BoardInfo for Board {

  fn step(&self) -> (String, Vec<usize>, f32) {
    let current_pos = self
        .field
        .iter()
        .fold("".to_string(), |acc, x| acc + &(x + 3).to_string()); //+3 to not border with +-
    let possible_actions = self.get_possible_moves();
    let controlled_fields = self.eval();
    let mut reward = (controlled_fields.0 as i32) - (controlled_fields.1 as i32);
    if !self.first_player_turn {
      reward *= -1;
    }
    (current_pos, possible_actions, reward as f32)
  }

  fn print_board(&self) {
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

  fn get_board_position(&self) -> [i8;36] {
    self.field
  }
  
  fn get_possible_moves(&self) -> Vec<usize> {
      if self.first_player_turn { self.first_player_moves.iter().copied().collect() } 
      else {self.second_player_moves.iter().copied().collect()}
  }

  fn get_possible_positions(&self) -> (Vec<String>, Vec<usize>) {
      let moves = self.get_possible_moves();
      let res = self
          .field
          .iter()
          .fold("".to_string(), |acc, x| acc + &(x + 3).to_string()); //+3 to not border with +-

      // prepare one entry for every new game position we can reach by one single, legal move
      let mut result_vector = vec![res; moves.len()];
      let player_val = if self.first_player_turn { 1 } else { -1 };

      for move_num in 0..moves.len() {
          //update the corresponding result position to the corresponding move
          let pos = moves[move_num];
          result_vector[move_num].replace_range(
              pos..(pos + 1),
              &(self.field[pos] + 3 + player_val).to_string(),
          );
      }
      (result_vector, moves)
    }
}


impl Board {
    pub fn get_board(rounds: u8) -> Board {
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
            total_rounds: rounds as usize,
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


    fn store_update(&mut self, pos: &usize, player_val: &i8) {
        self.field[*pos] += *player_val;
        self.flags[*pos] += *player_val;
        if self.field[*pos].abs() == 3 {
          // buildings on lv. 3 can't be upgraded
          self.second_player_moves.remove(pos);
          self.first_player_moves.remove(pos);
          return;
        }
        if *player_val == 1 {
          self.second_player_moves.remove(pos);
        } else {
          self.first_player_moves.remove(pos);
        }
    }

    pub fn try_move(&mut self, pos: usize) -> bool {
        let player_val = if self.first_player_turn { 1 } else { -1 };

        // check that field is not controlled by enemy, no enemy building on field, no own building on max lv (3) already exists
        if (self.first_player_turn && self.first_player_moves.contains(&pos)) ||
          (!self.first_player_turn && self.second_player_moves.contains(&pos)) 
        {
            self.store_update(&pos, &player_val);
            self.update_neighbours(pos, player_val);
            self.first_player_turn = !self.first_player_turn;
            return true;
        }
        self.print_board();
        let current_player = if self.first_player_turn { 1 } else { 2 };
        eprintln!("WARNING ILLEGAL MOVE {} BY PLAYER {}", pos, current_player);
        false // move wasn't allowed, do nothing
    }


    pub fn eval(&self) -> (u8, u8) {
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
    
    pub fn get_total_rounds(&self) -> usize {
        self.total_rounds
    }

}
