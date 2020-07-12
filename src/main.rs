use hello_rust::examples::xor;
use hello_rust::examples::mnist;
use hello_rust::examples::fortress;



fn main() -> Result<(),String>{

    //mnist::test_MNIST(); //not learning usefull results
    xor::test_xor(); 
    
    //fortress::test_fortress(); // more complex example, still needs some care


    Ok(())
}
