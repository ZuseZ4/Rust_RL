//! This submodule includes two interfaces for environments and agents.
//!
//! One environment and multiple agents are provided as example.
//! A training submodule is available for convenience.

/// This submodule contains the concrete agent interface and multiple agent examples.
pub mod agent;
mod algorithms;
/// This submodule contains the concrete env interface and a single example.
pub mod env;
/// This submodule offers some convenience functionality to simplify training agents.
pub mod training;
