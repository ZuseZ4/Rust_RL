mod agent_trait;
mod results;

mod ddql_agent;
mod dql_agent;
mod human_player;
mod ql_agent;
mod random_agent;

pub use agent_trait::Agent;

pub use ddql_agent::DDQLAgent;
pub use dql_agent::DQLAgent;
pub use human_player::HumanPlayer;
pub use ql_agent::QLAgent;
pub use random_agent::RandomAgent;
