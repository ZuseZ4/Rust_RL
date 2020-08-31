use crate::board::board_trait::BoardInfo;
use crate::engine::ql_engine::QLEngine;
use crate::engine::dql_engine::DQLEngine;
use crate::engine::engine_trait::Engine;
use crate::engine::random_engine::RandomEngine;
use crate::engine::human_player::HumanPlayer;
use crate::engine::gd_engine::GDEngine;

pub enum EngineType {
    R(RandomEngine),
    Q(QLEngine),
    D(DQLEngine),
    H(HumanPlayer),
    G(GDEngine),
}

impl EngineType {
    pub fn create_engine(
        rounds_per_game: u8,
        engine_number: u8,
        first_engine: bool,
    ) -> Result<EngineType, String> {
        match engine_number {
            1 => Ok(EngineType::R(RandomEngine::new(rounds_per_game, first_engine))),
            2 => Ok(EngineType::Q(QLEngine::new(rounds_per_game, first_engine, 1.))),// start with always exploring
            3 => Ok(EngineType::D(DQLEngine::new(rounds_per_game, first_engine,1.))),
            4 => Ok(EngineType::H(HumanPlayer::new(rounds_per_game, first_engine))),
            5 => Ok(EngineType::G(GDEngine::new(rounds_per_game, first_engine))),
            _ => Err(format!("Bad engine: {}", engine_number)),
        }
    }
}

impl Engine for EngineType {
    fn reset_board(&mut self) {
        match self {
            EngineType::R(r_engine) => r_engine.reset_board(),
            EngineType::Q(ql_engine) => ql_engine.reset_board(),
            EngineType::D(dql_engine) => dql_engine.reset_board(),
            EngineType::H(human_player) => human_player.reset_board(),
            EngineType::G(gd_engine) => gd_engine.reset_board(),
        }
    }
    fn get_id(&self) -> String {
        match self {
            EngineType::R(r_engine) => r_engine.get_id(),
            EngineType::Q(ql_engine) => ql_engine.get_id(),
            EngineType::D(dql_engine) => dql_engine.get_id(),
            EngineType::H(human_player) => human_player.get_id(),
            EngineType::G(gd_engine) => gd_engine.get_id(),
        }
    }
    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        match self {
            EngineType::R(r_engine) => r_engine.get_move(board),
            EngineType::Q(ql_engine) => ql_engine.get_move(board),
            EngineType::D(dql_engine) => dql_engine.get_move(board),
            EngineType::H(human_player) => human_player.get_move(board),
            EngineType::G(gd_engine) => gd_engine.get_move(board),
        }
    }
    fn finish_round(&mut self, result: i32) {
        match self {
            EngineType::R(r_engine) => r_engine.finish_round(result),
            EngineType::Q(ql_engine) => ql_engine.finish_round(result),
            EngineType::D(dql_engine) => dql_engine.finish_round(result),
            EngineType::H(human_player) => human_player.finish_round(result),
            EngineType::G(gd_engine) => gd_engine.finish_round(result),
        }
    }
    fn get_exploration_rate(&self) -> f32 {
      match self {
            EngineType::Q(ql_engine) => {return ql_engine.get_exploration_rate();},
            _ => {return 42.;},
      }
    }
    fn set_exploration_rate(&mut self, e: f32) -> Result<(),String> {
      match self {
            EngineType::Q(ql_engine) => {return ql_engine.set_exploration_rate(e);},
            _ => {return Ok(());},
      }
    }
}
