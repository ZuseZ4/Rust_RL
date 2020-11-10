#![deny(missing_docs)]

//! A crate including two submodules for neural networks and reinforcement learning.

/// A submodule including everything to build a neural network.
///
/// Currently multiple layers, error functions, and optimizers are provided.
pub mod network;

/// A submodule including everything to run reinforcement learning tasks.
///
/// This submodule includes two interfaces for environments and agents.
/// One environment and multiple agents are provided as example.
/// A training submodule is available for convenience.
pub mod rl;
