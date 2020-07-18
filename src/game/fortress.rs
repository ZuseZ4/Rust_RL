use std::cmp::Ordering;
use crate::board::smart_board;
use crate::engine::engine::EngineType;
use crate::engine::engine_trait::Engine;

pub struct Game {
    rounds: u8,
    res: (u32, u32, u32),
    last_result: i32,
    engine1: EngineType,
    engine2: EngineType,
}

impl Game {
    pub fn new(rounds_per_game: u8, game_type: u8) -> Result<Self, String> {
      // first digit encondes type of first engine, second digit the type of the second engine
      let first_engine = EngineType::create_engine(rounds_per_game, game_type/10, true)?; //first_player = true
      let second_engine = EngineType::create_engine(rounds_per_game, game_type%10, false)?; // first_player = false

      Ok(Game {
          rounds: rounds_per_game,
          res: (0, 0, 0),
          last_result: 42, //init value shouldn't be used
          engine1: first_engine,
          engine2: second_engine,
      })
    }

    fn update_results(&mut self, first_player_fields: u8, second_player_fields: u8) {
      match first_player_fields.cmp(&second_player_fields) {
        Ordering::Greater => {
          self.res.0 += 1;
          self.last_result = 1;   
        },
        Ordering::Less => {
          self.res.2 += 1;
          self.last_result = -1;
        },
        Ordering::Equal => {
          self.res.1 += 1;
          self.last_result = 0;
        },
      }
    }


    pub fn get_results(&self) -> (u32, u32, u32) {
        self.res
    }

    pub fn get_engine_ids(&self) -> (String, String) {
        (self.engine1.get_id(), self.engine2.get_id())
    }

    pub fn train(&mut self, num_games: u64) {
        self.play_games(num_games, true);
    }

    pub fn bench(&mut self, num_games: u64) -> (u32, u32, u32) {
        self.engine1.set_exploration_rate(0.).unwrap(); // exploration rate is in [0,1], so ignore error possibility
        self.engine2.set_exploration_rate(0.).unwrap();
        self.play_games(num_games, false)
    }

    fn play_games(&mut self, num_games: u64, train: bool) -> (u32, u32, u32) {
        self.res = (0, 0, 0);

        let mut board: smart_board::Board;
        for _game in 0..num_games {
            board = smart_board::Board::get_board(self.rounds);
            for _round in 0..board.get_total_rounds() {
              //TODO what if no move possible? 
              // parallelize

                board.try_move(self.engine1.get_move(&board));
                board.try_move(self.engine2.get_move(&board));
            }

            let game_res = board.eval(); // umstellen auf 1-hot encoded
            self.update_results(game_res.0, game_res.1);
            if train {
                self.engine1.finish_round(self.last_result);
                self.engine2.finish_round(self.last_result);
            }

            self.engine1.reset_board();
            self.engine2.reset_board();
        }
        self.res
    }
}
