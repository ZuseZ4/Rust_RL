use hello_rust::game::game;
use std::io;

fn main() -> Result<(),String>{
    // set number of rounds to play per game
    let mut engines = String::new();
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

    println!("please pick engines.");
    io::stdin()
        .read_line(&mut engines)
        .expect("Failed to read type of engines");

    let rounds: u8 = rounds.trim().parse().expect("please type a number");
    let training_games: u64 = training_games.trim().parse().expect("please type a number");
    let bench_games: u64 = bench_games.trim().parse().expect("please type a number");
    let engines: u8 = engines
        .trim()
        .parse()
        .expect("please type a number (11 for random-random, 22 for ai-ai, 33 for human-human");

    println!(
        "rounds: {}, #training games: {}, #bench games: {}\n",
        rounds, training_games, bench_games
    );

    let mut game = game::Game::new(rounds, engines)?;
    game.train(training_games);
    let res: (u32, u32, u32) = game.bench(bench_games);

    println!("engine1 ({}): {}", game.get_engine_ids().0, res.0);
    println!("draw: {}", res.1);
    println!("engine2 ({}): {}", game.get_engine_ids().1, res.2);
    Ok(())
}
