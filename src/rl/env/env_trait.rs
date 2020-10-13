use ndarray::{Array1, Array2};

/// This trait defines all functions on which agents and other user might depend.
pub trait Environment {
    /// Indicates that it's time to call env.reset()
    ///
    /// Done being True indicates that the episode has ended.
    /// This might mean that an agent lost his last life or the maximum number of rounds has been played.
    fn done(&self) -> bool;
    /// This returns an array containing all possible actions.
    ///
    /// The actual encoding depends on the environment, but it's expected to contain usize numbers which match actions in a reasonable way.
    fn get_legal_actions(&self) -> Array1<usize>;
    /// The central function which causes the environment to pass various information to the agent.
    ///
    /// The Array2 encodes the environment (the board).  
    /// The array1 encodes (il)-legal actions as (0) or 1.
    /// The last value returns a reward for the last action of the agent. 0 before the first action of the agent.
    fn step(&self) -> (Array2<f32>, Array1<f32>, f32);
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
    fn eval(&self) -> Vec<i8>;
}
