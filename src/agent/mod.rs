pub mod agent;
pub mod agent_trait;

pub mod random_agent;
pub mod human_player;
pub mod ql_agent;
pub mod dql_agent;
pub mod gd_agent;

pub use random_agent::RandomAgent;
pub use human_player::HumanPlayer;
pub use ql_agent::QLAgent;
pub use dql_agent::DQLAgent;
pub use gd_agent::GDAgent;
