mod env_trait;

mod fortress;
/// An implementation of the game Fortress from Gary Allen, 1984.
pub use fortress::Fortress;

mod tictactoe;
/// An implementation of the game TicTacToe.
pub use tictactoe::TicTacToe;

/// A trait defining the functions on which agents and other user depend.
pub use env_trait::Environment;
