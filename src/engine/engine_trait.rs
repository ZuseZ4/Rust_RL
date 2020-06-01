use crate::board::board_trait::BoardInfo;
pub trait Engine {
    fn get_id(&self) -> String;

    fn get_move(&mut self, board: &impl BoardInfo) -> usize;

    fn reset_board(&mut self);

    fn finish_round(&mut self, result: i32);
}
