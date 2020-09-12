use rust_rl::examples::{cifar10, fortress, mnist, xor};
use std::io;

fn main() -> Result<(), String> {
    let mut action = String::new();
    println!("Please enter 1/2/3/4 to run mnist/cifar/xor/fortress.");
    io::stdin()
        .read_line(&mut action)
        .expect("Failed to read number. please just enter 1,2,3, or 4!");
    let action: u8 = action.trim().parse().expect("please type a number");
    match action {
        1 => mnist::test_mnist(),
        2 => cifar10::test_cifar10(),
        3 => xor::test_xor(),
        4 => fortress::test_fortress(),
        _ => eprintln!("wrong number"),
    }
    Ok(())
}
