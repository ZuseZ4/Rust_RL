use crate::agent::agent::AgentType;
use crate::agent::agent_trait::Agent;
use crate::env::env::EnvType;
use crate::env::env_trait::Environment;
//use crate::env::fortress;

pub struct Trainer {
    //rounds: u8,
    env: EnvType,
    res: Vec<(u32, u32, u32)>,
    agents: Vec<AgentType>,
}

impl Trainer {
    pub fn new(env: EnvType, agents: Vec<AgentType>) -> Result<Self, String> {
        if agents.len() == 0 {
            return Err("At least one agent required!".to_string());
        }
        Ok(Trainer {
            //rounds: rounds_per_game,
            env,
            res: vec![(0, 0, 0); agents.len()],
            agents,
        })
    }

    pub fn get_results(&self) -> Vec<(u32, u32, u32)> {
        self.res.clone()
    }

    pub fn get_agent_ids(&self) -> Vec<String> {
        self.agents.iter().map(|a| a.get_id()).collect()
    }

    pub fn train(&mut self, num_games: u64) {
        self.play_games(num_games, true);
    }

    pub fn bench(&mut self, num_games: u64) -> Vec<(u32, u32, u32)> {
        self.agents
            .iter_mut()
            .for_each(|a| a.set_exploration_rate(0.).unwrap());
        self.play_games(num_games, false)
    }

    fn adjust_learning_rate(&mut self, sub_epoch_nr: u64, orig_learning_rate: Vec<f32>) {
        let exploration_rates: Vec<f32> = orig_learning_rate
            .iter()
            .map(|e| e * (10. - sub_epoch_nr as f32) / 10.)
            .collect();
        self.agents
            .iter_mut()
            .zip(exploration_rates.iter())
            .for_each(|(a, &e)| a.set_exploration_rate(e).unwrap());
        println!("New exploration rates: {:?}", exploration_rates);
    }

    fn update_results(&mut self, new_res: &Vec<i8>) {
        assert_eq!(
            new_res.len(),
            self.agents.len(),
            "results and number of agents differ!"
        );
        for (i, res) in new_res.iter().enumerate() {
            match res {
                -1 => self.res[i].0 += 1,
                0 => self.res[i].1 += 1,
                1 => self.res[i].2 += 1,
                _ => panic!("only allowed results are -1,0,1"),
            }
        }
    }

    fn play_games(&mut self, num_games: u64, train: bool) -> Vec<(u32, u32, u32)> {
        self.res = vec![(0, 0, 0); self.agents.len()];
        let sub_epoch: u64 = (num_games / 10) as u64;
        let orig_exploration_rates: Vec<f32> = self
            .agents
            .iter()
            .map(|a| a.get_exploration_rate())
            .collect();

        // parallelize
        for game in 0..num_games {
            self.env.reset();
            if (game % sub_epoch) == 0 && train {
                let sub_epoch_nr = game / sub_epoch;
                self.adjust_learning_rate(sub_epoch_nr, orig_exploration_rates.clone());
            }

            let mut done = false;
            loop {
                for agent in self.agents.iter_mut() {
                    if self.env.done() {
                        done = true;
                        break;
                    }
                    self.env.take_action(agent.get_move(&self.env));
                }
                if done {
                    break;
                }
            }

            let game_res = self.env.eval();
            self.update_results(&game_res);
            if train {
                for (i, agent) in self.agents.iter_mut().enumerate() {
                    agent.finish_round(game_res[i].into());
                }
            }
        }
        self.res.clone()
    }
}
