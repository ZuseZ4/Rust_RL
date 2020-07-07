use hello_rust::game::fortress::Game;
#[allow(unused_imports)]
use hello_rust::engine::{ai_engine, random_engine};

use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark_rand(c: &mut Criterion) {
    c.bench_function("random games", |b| b.iter(|| main_rand()));
}

pub fn criterion_benchmark_ai_rand(c: &mut Criterion) {
    c.bench_function("ai vs random games", |b| b.iter(|| main_ai_rand()));
}

pub fn criterion_benchmark_ai(c: &mut Criterion) {
    c.bench_function("ai vs ai games", |b| b.iter(|| main_ai()));
}

pub fn criterion_benchmark_train(c: &mut Criterion) {
    c.bench_function("train only", |b| b.iter(|| main_train()));
}

criterion_group!(benches, criterion_benchmark_rand, criterion_benchmark_ai_rand, criterion_benchmark_ai, criterion_benchmark_train);
criterion_main!(benches);

pub fn main_rand() -> Result<(),String> {
    let rounds: u8 = 25;
    let engines: u8 = 11;
    let bench_games: u64 = 100; 
    
    let mut game = Game::new(rounds, engines)?;
    game.bench(bench_games);
    Ok(())
}

pub fn main_ai_rand() -> Result<(),String> {
    let rounds: u8 = 25;
    let engines: u8 = 21;
    let train_games: u64 = 50;
    let bench_games: u64 = 100;

    let mut game = Game::new(rounds, engines)?;
    game.train(train_games);
    game.bench(bench_games);
    Ok(())
}

pub fn main_ai() -> Result<(),String> {
    let rounds: u8 = 25;
    let engines: u8 = 22;
    let train_games: u64 = 50;
    let bench_games: u64 = 100;

    let mut game = Game::new(rounds, engines)?;
    game.train(train_games);
    game.bench(bench_games);
    Ok(())
}

pub fn main_train() -> Result<(),String> {
    let rounds: u8 = 25;
    let engines: u8 = 22;
    let train_games: u64 = 50;

    let mut game = Game::new(rounds, engines)?;
    game.train(train_games);
    Ok(())
}
