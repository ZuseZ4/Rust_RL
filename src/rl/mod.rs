/// This submodule contains the concrete agent interface and multiple agent examples.
pub mod agent;

mod algorithms;

/// A submodule containing the Environment trait which all environments should implement.
///
/// An example implementation is given for the game "Fortress".
pub mod env;

/// This submodule offers some convenience functionality to simplify training agents.
pub mod training;
