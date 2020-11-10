use agent::*;
use env::fortress::Board;
use rust_rl::rl::{agent, env, training};
use std::io;
use training::Trainer;

pub fn main() {
    /*
    let mut auto_fill = String::new();
    println!("Run with default parameters? (1/0/2)");
    io::stdin()
        .read_line(&mut auto_fill)
        .expect("Failed to read 1 or 0 or 2");
    let auto_fill: usize = auto_fill.trim().parse().expect("please type 0 or 1 or 2");

    let params;
    if auto_fill == 1 {
        params = (25, 500, 1000, 12);
    } else if auto_fill == 2 {
        params = (25, parse_train_num(), 1000, 13);
    } else {
        params = parse_input();
    }*/
    let params = (25, 1_000, 1000, 12);

    let rounds = params.0;
    let training_games = params.1;
    let bench_games = params.2;
    let agents = params.3;
    let agent1 = get_agent(agents / 10, rounds, true).unwrap();
    let agent2 = get_agent(agents % 10, rounds, false).unwrap();

    let game = Board::new(rounds);

    let mut trainer = Trainer::new(Box::new(game), vec![agent1, agent2]).unwrap();
    trainer.train(training_games);

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

fn parse_train_num() -> u64 {
    let mut training_games = String::new();
    println!("please insert the number of training games.");
    io::stdin()
        .read_line(&mut training_games)
        .expect("Failed to read number of games");
    let training_games: u64 = training_games.trim().parse().expect("please type a number");
    training_games
}

fn parse_input() -> (u8, u64, u64, u8) {
    //set number of rounds to play per game
    let mut agents = String::new();
    let mut rounds = String::new();
    let mut training_games = String::new();
    let mut bench_games = String::new();
    println!("please insert the number of rounds per game.");
    io::stdin()
        .read_line(&mut rounds)
        .expect("Failed to read number of rounds");

    println!("please insert the number of training games.");
    io::stdin()
        .read_line(&mut training_games)
        .expect("Failed to read number of games");

    println!("please insert the number of benchmark games.");
    io::stdin()
        .read_line(&mut bench_games)
        .expect("Failed to read number of games");

    println!("please pick agents.");
    io::stdin()
        .read_line(&mut agents)
        .expect("Failed to read type of agents");

    let rounds: u8 = rounds.trim().parse().expect("please type a number");
    let training_games: u64 = training_games.trim().parse().expect("please type a number");
    let bench_games: u64 = bench_games.trim().parse().expect("please type a number");
    let agents: u8 = agents
        .trim()
        .parse()
        .expect("please type a number (11 for random-random, 22 for ai-ai, 33 for human-human");

    println!(
        "rounds: {}, #training games: {}, #bench games: {}\n",
        rounds, training_games, bench_games
    );
    (rounds, training_games, bench_games, agents)
}

fn get_agent(agent_num: u8, rounds: u8, first_player: bool) -> Result<Box<dyn Agent>, String> {
    match agent_num {
        1 => Ok(Box::new(DQLAgent::new(1.))),
        2 => Ok(Box::new(QLAgent::new(1.))),
        3 => Ok(Box::new(RandomAgent::new())),
        4 => Ok(Box::new(HumanPlayer::new())),
        _ => Err("Only implemented agents 1-4!".to_string()),
    }
}
