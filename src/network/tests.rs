
#[cfg(test)]
mod tests {
    use crate::network::nn::NeuralNetwork;
    use ndarray::{array, Array1, Array2};
    use rand::Rng;

    fn new(i_dim: usize) -> NeuralNetwork {
      let mut nn = NeuralNetwork::new(i_dim);
      nn.add_connection("dense", 4); //Dense with 2 / 3 output neurons
      nn.add_activation("sigmoid"); //Sigmoid
      nn.add_connection("dense", 1); //Dense with 1 output neuron
      nn.add_activation("sigmoid"); //Sigmoid
      nn
    }

    fn test(mut nn: NeuralNetwork, input: Array2<f32>, fb: Array2<f32>) {
      let mut current_input;
      let mut current_fb;
      let mut diff;
      for i in 0..input.nrows() {
        current_input = input.row(i).into_owned().clone();
        current_fb = fb.row(i).into_owned().clone();
        diff = (&nn.forward(current_input) - &current_fb).sum().abs();

        assert!(diff < 0.2, "failed learning: {}", diff);
      }
    }

    fn train(nn: &mut NeuralNetwork, num: usize, input: &Array2<f32>, fb: &Array2<f32>) {
      let mut pos;
      let mut current_input;
      let mut current_fb;
      for _ in 0..num {
        pos = rand::thread_rng().gen_range(0, input.nrows()) as usize;
        current_input = input.row(pos).into_owned().clone();
        current_fb = fb.row(pos).into_owned().clone();
        nn.forward(current_input);
        nn.backward(current_fb);
      }
    }

    #[test]
    fn test_and() {
      let input = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.],[1.,1.],[1.,1.]]; // AND
      let fb = array![[0.],[0.],[0.],[1.],[1.],[1.]]; //AND work ok with 200k examples (10 and 01 are classified correctly, but close to 0.5)
      let mut nn = new(2);
      train(&mut nn, 40000, &input, &fb);
      test(nn, input, fb);
    }

    #[test]
    fn test_or() {
      let input = array![[0.,0.],[0.,0.],[0.,0.],[0.,1.],[1.,0.],[1.,1.]]; // OR
      let fb = array![[0.],[0.],[0.],[1.],[1.],[1.]];//OR works great with 200k examples
      let mut nn = new(2);
      train(&mut nn, 40000, &input, &fb);
      test(nn, input, fb);
    }

    #[test]
    fn test_not() {
      let input = array![[0.],[1.]];
      let fb = array![[1.],[0.]];// NOT works great with 200k examples
      let mut nn = new(1);
      train(&mut nn, 40000, &input, &fb);
      test(nn, input, fb);
    }


    #[test]
    fn test_first() {
      let input = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.]]; // FIRST
      let fb = array![[0.],[0.],[1.],[1.]]; //First works good with 200k examples
      let mut nn = new(2);
      train(&mut nn, 40000, &input, &fb);
      test(nn, input, fb);
    }


    #[test]
    fn test_xor() {
      let input = array![[0.,0.],[0.,1.],[1.,0.],[1.,1.]];
      let fb = array![[0.],[1.],[1.],[0.]];//XOR
      let mut nn = new(2);
      train(&mut nn, 100000, &input, &fb);
      test(nn, input, fb);
    }

}
