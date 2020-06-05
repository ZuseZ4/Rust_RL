use crate::board::board_trait::BoardInfo;
use crate::engine::ai_engine::AIEngine;
use crate::engine::engine_trait::Engine;
use crate::engine::random_engine::RandomEngine;
use crate::engine::human_player::HumanPlayer;
use crate::engine::gd_engine::gd_engine::GDEngine;

pub enum EngineType {
    R(RandomEngine),
    A(AIEngine),
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
            2 => Ok(EngineType::A(AIEngine::new(rounds_per_game, first_engine, 0.2))),
            3 => Ok(EngineType::H(HumanPlayer::new(rounds_per_game, first_engine))),
            4 => Ok(EngineType::G(GDEngine::new(rounds_per_game, first_engine))),
            _ => Err(format!("Bad engine: {}", engine_number)),
        }
    }
}

impl Engine for EngineType {
    fn reset_board(&mut self) {
        match self {
            EngineType::R(r_engine) => r_engine.reset_board(),
            EngineType::A(ai_engine) => ai_engine.reset_board(),
            EngineType::H(human_player) => human_player.reset_board(),
            EngineType::G(gd_engine) => gd_engine.reset_board(),
        }
    }
    fn get_id(&self) -> String {
        match self {
            EngineType::R(r_engine) => r_engine.get_id(),
            EngineType::A(ai_engine) => ai_engine.get_id(),
            EngineType::H(human_player) => human_player.get_id(),
            EngineType::G(gd_engine) => gd_engine.get_id(),
        }
    }
    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        match self {
            EngineType::R(r_engine) => r_engine.get_move(board),
            EngineType::A(ai_engine) => ai_engine.get_move(board),
            EngineType::H(human_player) => human_player.get_move(board),
            EngineType::G(gd_engine) => gd_engine.get_move(board),
        }
    }
    fn finish_round(&mut self, result: i32) {
        match self {
            EngineType::R(r_engine) => r_engine.finish_round(result),
            EngineType::A(ai_engine) => ai_engine.finish_round(result),
            EngineType::H(human_player) => human_player.finish_round(result),
            EngineType::G(gd_engine) => gd_engine.finish_round(result),
        }
    }
}
