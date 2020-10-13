//! A submodule containing the Environment trait which all future environments should implement.
//!
//! An example implementation is given for the game "Fortress".

mod env_trait;

/// An implementation of the game Fortress from Gary Allen, 1984.
pub mod fortress;

/// A trait defining the functions on which agents and other user depend.
pub use env_trait::Environment;
