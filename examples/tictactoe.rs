use agent::*;
use env::TicTacToe;
use rust_rl::network::nn::NeuralNetwork;
use rust_rl::rl::{agent, env, training};
use std::io;
use training::{utils, Trainer};

fn new(learning_rate: f32, batch_size: usize) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((1, 3, 3), "mse".to_string(), "adam".to_string());
    nn.set_batch_size(batch_size);
    nn.set_learning_rate(learning_rate);
    nn.add_flatten();
    nn.add_dense(100);
    nn.add_activation("sigmoid");
    nn.add_dense(9);
    //nn.add_activation("sigmoid");
    nn.print_setup();
    nn
}

pub fn main() {
    let mut auto_fill = String::new();
    println!("Run with default parameters? (y/n)");
    io::stdin()
        .read_line(&mut auto_fill)
        .expect("Failed to read y or n!");
    let auto_fill: String = auto_fill.trim().parse().expect("Please type y or n!");

    let ((train_games, bench_games, iterations), agent_numbers) = match auto_fill.as_str() {
        "y" => ((40_000, 10_000, 5), vec![0, 2]),
        "n" => (utils::read_game_numbers(), utils::read_agents(2)),
        _ => panic!("please only answer y or n!"),
    };

    let agents = get_agents(agent_numbers).unwrap();
    let game = TicTacToe::new();

    let mut trainer = Trainer::new(Box::new(game), agents, true).unwrap();
    trainer.train_bench_loops(train_games, bench_games, iterations);
}

fn get_agents(agent_nums: Vec<usize>) -> Result<Vec<Box<dyn Agent>>, String> {
    let mut res: Vec<Box<dyn Agent>> = vec![];
    let batch_size = 8;
    for agent_num in agent_nums {
        let new_agent: Result<Box<dyn Agent>, String> = match agent_num {
            0 => Ok(Box::new(DDQLAgent::new(
                1.,
                batch_size,
                new(3e-4, batch_size),
            ))),
            1 => Ok(Box::new(DQLAgent::new(
                1.,
                batch_size,
                new(0.0003, batch_size),
            ))),
            2 => Ok(Box::new(QLAgent::new(1., 0.1, 3 * 3))),
            3 => Ok(Box::new(RandomAgent::new())),
            4 => Ok(Box::new(HumanPlayer::new())),
            _ => Err("Only implemented agents 1-4!".to_string()),
        };
        res.push(new_agent?);
    }
    Ok(res)
}
