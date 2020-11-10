mod env_trait;

/// An implementation of the game Fortress from Gary Allen, 1984.
pub mod fortress;

/// A trait defining the functions on which agents and other user depend.
pub use env_trait::Environment;
