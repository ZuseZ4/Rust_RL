use ndarray::{Array1, Array2};

/// This trait defines all functions on which agents and other user might depend.
pub trait Environment {
    /// The central function which causes the environment to pass various information to the agent.
    ///
    /// The Array2 encodes the environment (the board).  
    /// The array1 encodes actions as true (allowed) or false (illegal).
    /// The third value returns a reward for the last action of the agent. 0 before the first action of the agent.
    /// The final bool value (done) indicates, wether it is time to reset the environment.
    fn step(&self) -> (Array2<f32>, Array1<bool>, f32, bool);
    /// Update the environment based on the action given.
    ///
    /// If the action is allowed for the currently active agent then update the environment and return true.
    /// Otherwise do nothing and return false. The same agent can then try a new move.
    fn take_action(&mut self, action: usize) -> bool;
    /// Shows the current envrionment state in a graphical way.
    ///
    /// The representation is environment specific and might be either by terminal, or in an extra window.
    fn render(&self);
    /// Resets the environment to the initial state.
    fn reset(&mut self);
    /// A vector with one entry for each agent, either 1 (agent won), 0 (draw), or -1 (agent lost).
    fn eval(&mut self) -> Vec<i8>;
}
