use crate::rl::agent::Agent;
use crate::rl::env::Environment;
use ndarray::Array2;

/// A trainer works on a given environment and a set of agents.
pub struct Trainer {
    env: Box<dyn Environment>,
    res: Vec<(u32, u32, u32)>,
    agents: Vec<Box<dyn Agent>>,
    print: bool,
}

impl Trainer {
    /// We construct a Trainer by passing a single environment and one or more (possibly different) agents.
    pub fn new(
        env: Box<dyn Environment>,
        agents: Vec<Box<dyn Agent>>,
        print: bool,
    ) -> Result<Self, String> {
        if agents.is_empty() {
            return Err("At least one agent required!".to_string());
        }
        Ok(Trainer {
            env,
            res: vec![(0, 0, 0); agents.len()],
            agents,
            print,
        })
    }

    /// Returns a (#won, #draw, #lost) tripple for each agent.
    ///
    /// The numbers are accumulated over all train and bench games, either since the beginning, or the last reset_results() call.
    pub fn get_results(&self) -> Vec<(u32, u32, u32)> {
        self.res.clone()
    }

    /// Resets the (#won, #draw, #lost) values for each agents to (0,0,0).
    pub fn reset_results(&mut self) {
        self.res = vec![(0, 0, 0); self.agents.len()];
    }

    /// Returns a Vector containing the string identifier of each agent.
    pub fn get_agent_ids(&self) -> Vec<String> {
        self.agents.iter().map(|a| a.get_id()).collect()
    }

    /// Executes n training games folowed by m bench games.
    /// Repeates this cycle for i iterations.
    pub fn train_bench_loops(&mut self, n: u64, m: u64, i: u64) {
        for _ in 0..i {
            self.train(n);
            let res: Vec<(u32, u32, u32)> = self.bench(m);
            if self.print {
                for (agent, result) in res.iter().enumerate() {
                    println!(
                        "agent{} ({}): lost: {}, draw: {}, won: {}",
                        agent,
                        self.get_agent_ids()[agent],
                        result.0,
                        result.1,
                        result.2
                    );
                }
            }
        }
    }

    /// Executes the given amount of (independent) training games.
    ///
    /// Results are stored and agents are expected to update their internal parameters   
    /// in order to adjust to the game and performe better on subsequent games.
    pub fn train(&mut self, num_games: u64) {
        self.play_games(num_games, true);
    }

    /// Allows changing the learning rate of the agents.
    pub fn set_learning_rates(&mut self, new_vals: &[f32]) -> Result<(), String> {
        assert_eq!(
            new_vals.len(),
            self.agents.len(),
            "wrong length of input slice!"
        );
        for (agent, new_val) in self.agents.iter_mut().zip(new_vals.iter()) {
            agent.set_learning_rate(*new_val)?;
        }
        Ok(())
    }

    /// Allows changing the exploration rate of the agents.
    pub fn set_exploration_rates(&mut self, new_vals: &[f32]) -> Result<(), String> {
        assert_eq!(
            new_vals.len(),
            self.agents.len(),
            "wrong length of input slice!"
        );
        for (agent, new_val) in self.agents.iter_mut().zip(new_vals.iter()) {
            agent.set_exploration_rate(*new_val)?;
        }
        Ok(())
    }

    /// Executes the given amount of (independent) bench games.
    ///
    /// Results are stored and agents are expected to not learn based on bench games.
    pub fn bench(&mut self, num_games: u64) -> Vec<(u32, u32, u32)> {
        self.play_games(num_games, false)
    }

    fn update_results(&mut self, new_res: &[i8]) {
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
        for agent in self.agents.iter_mut() {
            agent.freeze(!train);
        }
        self.res = vec![(0, 0, 0); self.agents.len()];

        // TODO parallelize
        println!("num games: {}", num_games);
        for _game in 0..num_games {
            self.env.reset();

            let final_state: Array2<f32> = 'outer: loop {
                for agent in self.agents.iter_mut() {
                    let (env, actions, reward, done) = self.env.step();
                    if done {
                        break 'outer env;
                    }
                    let res = self.env.take_action(agent.get_move(env, actions, reward));
                    if !res {
                        println!("illegal move!");
                    }
                }
            };

            let game_res = self.env.eval();
            self.update_results(&game_res);
            if train {
                for (i, agent) in self.agents.iter_mut().enumerate() {
                    agent.finish_round(game_res[i], final_state.clone());
                }
            }
        }
        self.res.clone()
    }
}
