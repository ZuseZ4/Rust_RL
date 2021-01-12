mod agent_trait;

mod dql_agent;
mod ddql_agent;
mod human_player;
mod ql_agent;
mod random_agent;

pub use agent_trait::Agent;

pub use dql_agent::DQLAgent;
pub use ddql_agent::DDQLAgent;
pub use human_player::HumanPlayer;
pub use ql_agent::QLAgent;
pub use random_agent::RandomAgent;
