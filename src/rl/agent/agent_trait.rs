use crate::rl::env::Environment;

/// A trait including all functions required to train them.
pub trait Agent {

    /// Returns a simple string identifying the specific agent type.
    fn get_id(&self) -> String;

    /// Expect the agent to return a single usize value corresponding to a (legal) action he picked.
    ///
    /// The concrete encoding of actions as usize value has to be looked up in the documentation of the specific environment.  
    /// Advanced agents shouldn't need knowledge about the used encoding.
    fn get_move(&mut self, board: &Box<dyn Environment>) -> usize;

    /// Informs the agent that the current epoch has finished and tells him about his final result.
    ///
    /// Common values might be -1/0/1 if the agent achieved a loss/draw/win.
    fn finish_round(&mut self, result: i32);

    /// Updates the exploration rate if it lies in the range [0,1].
    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String>;

    /// Returns the current exploration rate.
    fn get_exploration_rate(&self) -> f32;
}
