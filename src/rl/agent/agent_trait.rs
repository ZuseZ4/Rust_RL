use crate::rl::env::Environment;
pub trait Agent {
    fn get_id(&self) -> String;

    fn get_move(&mut self, board: &Box<dyn Environment>) -> usize;

    fn finish_round(&mut self, result: i32);

    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String>;

    fn get_exploration_rate(&self) -> f32;
}
