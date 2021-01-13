use std::io;

/// A helper function to create agents based on terminal input.
pub fn read_agents(n: usize) -> Vec<usize> {
    let mut agents: Vec<usize> = vec![];

    println!(
        "\nPlease insert {} numbers, seperated by whitespace, to select the agents.",
        n
    );
    println!("(0 for ddql, 1 for dql, 2 for ql, 3 for random, 4 for human)");
    let stdin = io::stdin();
    loop {
        let mut buffer = String::new();
        stdin.read_line(&mut buffer).unwrap();
        let nums: Vec<&str> = buffer.split(' ').collect();
        if nums.len() != n {
            println!("Please enter exactly {} values", n);
            continue;
        }
        for agent_num in nums
            .iter()
            .map(|num| usize::from_str_radix(num.trim(), 10).unwrap())
        {
            agents.push(agent_num);
        }
        break;
    }
    agents
}

/// Reads the amount of training- and test-games from terminal.
pub fn read_game_numbers() -> (u64, u64, u64) {
    loop {
        println!("\nPlease enter #training_games #test_games #iterations, seperated by whitespace");
        let stdin = io::stdin();
        let mut buffer = String::new();
        stdin.read_line(&mut buffer).unwrap();
        let nums: Vec<&str> = buffer.split(' ').collect();
        if nums.len() != 3 {
            println!("Please enter exactly three values");
            continue;
        }
        let nums: Vec<u64> = nums
            .iter()
            .map(|num| u64::from_str_radix(num.trim(), 10).unwrap())
            .collect();
        return (nums[0], nums[1], nums[2]);
    }
}

/// For round based games, reads an usize value from terminal.
pub fn read_rounds_per_game() -> usize {
    //set number of rounds to play per game
    let mut rounds = String::new();
    println!("please insert the number of rounds per game.");
    io::stdin()
        .read_line(&mut rounds)
        .expect("Failed to read number of rounds");

    let rounds: usize = rounds.trim().parse().expect("please type a number");
    rounds
}
