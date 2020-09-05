use crate::game::fortress;
use std::io;

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

pub fn test_fortress() {
    let mut auto_fill = String::new();
    println!("Run with default parameters? (1/0/2)");
    io::stdin()
        .read_line(&mut auto_fill)
        .expect("Failed to read 1 or 0 or 2");
    let auto_fill: usize = auto_fill.trim().parse().expect("please type 0 or 1 or 2");

    let params;
    if auto_fill == 1 {
        params = (25, 4_000, 1000, 13);
    } else if auto_fill == 2 {
        params = (25, parse_train_num(), 1000, 13);
    } else {
        params = parse_input();
    }

    let rounds = params.0;
    let training_games = params.1;
    let bench_games = params.2;
    let agents = params.3;

    let mut game = fortress::Game::new(rounds, agents).unwrap();
    game.train(training_games);

    let res: (u32, u32, u32) = game.bench(bench_games);

    println!("agent1 ({}): {}", game.get_agent_ids().0, res.0);
    println!("draw: {}", res.1);
    println!("agent2 ({}): {}", game.get_agent_ids().1, res.2);
}
