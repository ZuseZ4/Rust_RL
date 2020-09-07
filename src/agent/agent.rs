use crate::agent::agent_trait::Agent;
use crate::agent::DQLAgent;
use crate::agent::HumanPlayer;
use crate::agent::QLAgent;
use crate::agent::RandomAgent;
use crate::env::env_trait::Environment;

pub enum AgentType {
    R(RandomAgent),
    Q(QLAgent),
    D(DQLAgent),
    H(HumanPlayer),
}

impl AgentType {
    pub fn create_agent(
        rounds_per_game: u8,
        agent_number: u8,
        first_agent: bool,
    ) -> Result<AgentType, String> {
        match agent_number {
            1 => Ok(AgentType::R(RandomAgent::new(rounds_per_game, first_agent))),
            2 => Ok(AgentType::Q(QLAgent::new(rounds_per_game, first_agent, 1.))), // start with always exploring
            3 => Ok(AgentType::D(DQLAgent::new(
                rounds_per_game,
                first_agent,
                1.,
            ))),
            4 => Ok(AgentType::H(HumanPlayer::new(rounds_per_game, first_agent))),
            _ => Err(format!("Bad agent: {}", agent_number)),
        }
    }
}

impl Agent for AgentType {
    fn get_id(&self) -> String {
        match self {
            AgentType::R(r_agent) => r_agent.get_id(),
            AgentType::Q(ql_agent) => ql_agent.get_id(),
            AgentType::D(dql_agent) => dql_agent.get_id(),
            AgentType::H(human_player) => human_player.get_id(),
        }
    }
    fn get_move(&mut self, board: &impl Environment) -> usize {
        match self {
            AgentType::R(r_agent) => r_agent.get_move(board),
            AgentType::Q(ql_agent) => ql_agent.get_move(board),
            AgentType::D(dql_agent) => dql_agent.get_move(board),
            AgentType::H(human_player) => human_player.get_move(board),
        }
    }
    fn finish_round(&mut self, result: i32) {
        match self {
            AgentType::R(r_agent) => r_agent.finish_round(result),
            AgentType::Q(ql_agent) => ql_agent.finish_round(result),
            AgentType::D(dql_agent) => dql_agent.finish_round(result),
            AgentType::H(human_player) => human_player.finish_round(result),
        }
    }
    fn get_exploration_rate(&self) -> f32 {
        match self {
            AgentType::Q(ql_agent) => {
                return ql_agent.get_exploration_rate();
            }
            AgentType::D(dql_agent) => {
                return dql_agent.get_exploration_rate();
            }
            _ => {
                return 42.;
            }
        }
    }
    fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        match self {
            AgentType::Q(ql_agent) => {
                return ql_agent.set_exploration_rate(e);
            }
            AgentType::D(dql_agent) => {
                return dql_agent.set_exploration_rate(e);
            }
            _ => {
                return Ok(());
            }
        }
    }
}
