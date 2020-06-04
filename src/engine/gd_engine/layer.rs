use crate::board::board_trait::BoardInfo;
use crate::engine::ai_engine::AILayer;
use crate::engine::engine_trait::Layer;
use crate::engine::random_engine::RandomLayer;
use crate::engine::human_player::HumanPlayer;

pub enum LayerType {
    S(SoftmaxLayer),
    D(DenseLayer),
    //R(RandomLayer),
    //A(AILayer),
    //H(HumanPlayer),
}

impl LayerType {
    pub fn create_engine(
        layer_number: u8,
    ) -> Result<LayerType, String> {
        match engine_number {
            1 => Ok(LayerType::D(DenseLayer::new())),
            2 => Ok(LayerType::S(SoftmaxLayer::new())),
            _ => Err(format!("Bad Layer: {}", layer_number)),
        }
    }
}

impl Layer for LayerType {
    fn get_id(&self) -> String {
        match self {
            LayerType::R(r_engine) => r_engine.get_id(),
            LayerType::A(ai_engine) => ai_engine.get_id(),
        }
    }
    fn get_move(&mut self, board: &impl BoardInfo) -> usize {
        match self {
            LayerType::R(r_engine) => r_engine.get_move(board),
            LayerType::A(ai_engine) => ai_engine.get_move(board),
        }
    }
    fn finish_round(&mut self, result: i32) {
        match self {
            LayerType::R(r_engine) => r_engine.finish_round(result),
            LayerType::A(ai_engine) => ai_engine.finish_round(result),
        }
    }
}
