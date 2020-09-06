#[allow(unused_imports)]
use rust_rl::examples::{cifar10, fortress, mnist, xor};

fn main() -> Result<(), String> {
    mnist::test_mnist();
    //cifar10::test_cifar10();
    //xor::test_xor();
    //fortress::test_fortress();

    Ok(())
}
