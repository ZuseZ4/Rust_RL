use crate::env::env_trait::Environment;
use crate::env::fortress::Board;
use ndarray::{Array1, Array2};

pub enum EnvType {
    F(Board),
}

impl EnvType {
    pub fn create_env(rounds_per_game: u8, env_number: usize) -> Result<EnvType, String> {
        match env_number {
            1 => Ok(EnvType::F(Board::new(rounds_per_game))),
            _ => Err(format!("Bad env: {}", env_number)),
        }
    }
}

impl Environment for EnvType {
    fn done(&self) -> bool {
        match self {
            EnvType::F(fortress_env) => fortress_env.done(),
        }
    }
    fn get_legal_actions(&self) -> Array1<usize> {
        match self {
            EnvType::F(fortress_env) => fortress_env.get_legal_actions(),
        }
    }
    fn step(&self) -> (Array2<f32>, Array1<f32>, f32) {
        match self {
            EnvType::F(fortress_env) => fortress_env.step(),
        }
    }
    fn take_action(&mut self, action: usize) -> bool {
        match self {
            EnvType::F(fortress_env) => fortress_env.take_action(action),
        }
    }
    fn render(&self) {
        match self {
            EnvType::F(fortress_env) => fortress_env.render(),
        }
    }
    fn reset(&mut self) {
        match self {
            EnvType::F(fortress_env) => fortress_env.reset(),
        }
    }
    fn eval(&self) -> Vec<i8> {
        match self {
            EnvType::F(fortress_env) => fortress_env.eval(),
        }
    }
}
