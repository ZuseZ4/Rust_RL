use agent::*;
use env::Fortress;
use rust_rl::network::nn::NeuralNetwork;
use rust_rl::rl::{agent, env, training};
use std::io;
use training::{utils, Trainer};

fn new(learning_rate: f32, batch_size: usize) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((1, 6, 6), "mse".to_string(), "adam".to_string());
    nn.set_batch_size(batch_size);
    nn.set_learning_rate(learning_rate);
    //nn.add_convolution((3, 3), 32, 0);
    //nn.add_activation("sigmoid");
    nn.add_flatten();
    nn.add_dense(100);
    nn.add_activation("sigmoid");
    nn.add_dense(50);
    nn.add_activation("sigmoid");
    nn.add_dense(36);
    nn.print_setup();
    nn
}

pub fn main() {
    let mut auto_fill = String::new();
    println!("Run with default parameters? (Y/n)");
    io::stdin()
        .read_line(&mut auto_fill)
        .expect("Failed to read y or no!");
    let auto_fill: String = auto_fill.trim().parse().expect("Please type y or no!");

    let (rounds, agent_numbers) = match auto_fill.as_str() {
        "y" | "Y" | "" => (15, vec![3, 0]),
        "n" => (utils::read_rounds_per_game(), utils::read_agents(2)),
        _ => panic!("please only answer y or n!"),
    };
    let agents = get_agents(agent_numbers).unwrap();

    let game = Fortress::new(rounds as usize);

    let mut trainer = Trainer::new(Box::new(game), agents, true).unwrap();

    trainer.train(10_000);
    println!("lost/draw/won: {:?}", trainer.bench(10000)[1]);
    //trainer.set_learning_rates(&vec![1e-4, 1e-4]);
    trainer.set_exploration_rates(&vec![0.1, 0.1]);

    trainer.train(20_000);
    println!("lost/draw/won: {:?}", trainer.bench(10000)[1]);
    //trainer.set_learning_rates(&vec![1e-5, 1e-5]);
    trainer.set_exploration_rates(&vec![0.01, 0.01]);

    trainer.train(60_000);
    println!("lost/draw/won: {:?}", trainer.bench(10000)[1]);
}

fn get_agents(agent_nums: Vec<usize>) -> Result<Vec<Box<dyn Agent>>, String> {
    let mut res: Vec<Box<dyn Agent>> = vec![];
    let batch_size = 16;
    for agent_num in agent_nums {
        let new_agent: Result<Box<dyn Agent>, String> = match agent_num {
            0 => Ok(Box::new(DDQLAgent::new(
                1.,
                batch_size,
                new(1e-4, batch_size),
            ))),
            1 => Ok(Box::new(DQLAgent::new(
                1.,
                batch_size,
                new(0.001, batch_size),
            ))),
            2 => Ok(Box::new(QLAgent::new(1., 0.1, 6 * 6))),
            3 => Ok(Box::new(RandomAgent::new())),
            4 => Ok(Box::new(HumanPlayer::new())),
            _ => Err("Only implemented agents 1-4!".to_string()),
        };
        res.push(new_agent?);
    }
    Ok(res)
}
