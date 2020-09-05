#[allow(unused_imports)]
use hello_rust::examples::{cifar10, fortress, mnist, xor};

fn main() -> Result<(), String> {
    mnist::test_MNIST(); //not learning usefull results
                         //cifar10::test_Cifar10(); //not learning usefull results
                         //xor::test_xor();

    //fortress::test_fortress(); // more complex example, still needs some care

    Ok(())
}
