use env::TicTacToe;
use rust_rl::rl::{env, training};
use std::io;
use training::{utils, Trainer};

pub fn main() {
    let mut auto_fill = String::new();
    println!("Run with default parameters? (y/n)");
    io::stdin()
        .read_line(&mut auto_fill)
        .expect("Failed to read y or no!");
    let auto_fill: String = auto_fill.trim().parse().expect("Please type y or no!");

    let ((train_games, bench_games), agents) = match auto_fill.as_str() {
        "y" => ((5000, 1000), utils::get_agents(vec![1, 2]).unwrap()),
        "n" => (utils::read_game_numbers(), utils::read_agents(2).unwrap()),
        _ => panic!("please only answer y or n!"),
    };

    let game = TicTacToe::new();

    let mut trainer = Trainer::new(Box::new(game), agents).unwrap();
    trainer.train(train_games);

    let res: Vec<(u32, u32, u32)> = trainer.bench(bench_games);

    println!(
        "agent1 ({}): lost: {}, draw: {}, won: {}",
        trainer.get_agent_ids()[0],
        res[0].0,
        res[0].1,
        res[0].2
    );
    println!(
        "agent2 ({}): lost: {}, draw: {}, won: {}",
        trainer.get_agent_ids()[1],
        res[1].0,
        res[1].1,
        res[1].2
    );
}
