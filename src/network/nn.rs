use ndarray::{Array, Array1};
use crate::network::layer::LayerType;
use crate::network::layer_trait::Layer;

pub struct HyperParameter {
  batch_size: usize,
  learning_rate: f32,
  _gamma: f32,
  _decay_rate: f32,
  _resume: bool,
  _render: bool,
}
impl HyperParameter {
  pub fn new() -> Self {
    HyperParameter{
      batch_size: 100,
      learning_rate: 1e-1, //10e-4
      _gamma: 0.99,
      _decay_rate: 0.99,
      _resume: false,
      _render: false,
    }
  }
}


pub struct NeuralNetwork {
  input_dim: usize,
  ll_output_dim: usize,
  h_p: HyperParameter,
  layers: Vec<LayerType>,
  last_input:  Array1<f32>,
  last_output: Array1<f32>,
  last_target: Array1<f32>,
}


impl NeuralNetwork {
  pub fn new(input_dim: usize) -> NeuralNetwork {

    NeuralNetwork{
      input_dim,
      ll_output_dim: input_dim,
      layers:  vec![],
      h_p: HyperParameter::new(),
      last_input:  Array::zeros(input_dim),
      last_output: Array::zeros(input_dim),
      last_target: Array::zeros(input_dim),
    }

    //self.add_layer("activation", 1); //Softmax

    //self.add_connection("dense", 2); //Dense with 2 output neurons
    //self.add_activation("sigmoid"); //Sigmoid
    //self.add_connection("dense", 1); //Dense with 1 output neuron
    //self.add_activation("sigmoid"); //Sigmoid

  }

  pub fn add_activation(&mut self, layer_kind: &str) {
    match layer_kind {
      "softmax" => self.layers.push(LayerType::new_activation(1).unwrap()),
      "sigmoid" => self.layers.push(LayerType::new_activation(2).unwrap()),
      _ => { println!("unknown activation function. Doing nothing!"); return;},
    }
    // don't change output dim, activation layers don't change dimensions
  }

  pub fn add_connection(&mut self, layer_kind: &str, output_dim: usize) {
    match layer_kind {
      "dense" => self.layers.push(LayerType::new_connection(1, self.ll_output_dim, output_dim, self.h_p.batch_size, self.h_p.learning_rate).unwrap()),
      _ => println!("unknown type, use \"connection\" or \"activation\". Doing nothing!"),
    }
    self.ll_output_dim = output_dim; // update value to output size of new last layer
  }
}

fn normalize(x: Array1<f32>) -> Array1<f32> {
  x.map(|&x| (x+3.0)/6.0)
}

impl NeuralNetwork {

  pub fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
    let mut input = x;//normalize(x);
    self.last_input = input.clone();
    for i in 0..self.layers.len() {
      input = self.layers[i].forward(input);
    }
    self.last_output = input; //output
    self.last_output.clone()
  }

  pub fn backward(&mut self, target: Array1<f32>) {
    self.last_target = target.clone();
    let mut fb = &self.last_output - &target; //should be correct
    for i in (0..self.layers.len()).rev() {
      fb = self.layers[i].backward(fb);
    }
  }

  pub fn error(&mut self) {
    let mse = self.last_output.iter()
      .zip(self.last_target.iter())
      .fold(0.0, |sum, (&x, &y)| sum + 0.5 * (x-y).powf(2.0));
    println!("input: {}, expected output: {}, was: {}", self.last_input, self.last_target, self.last_output);
    println!("MSE: {}",mse);
  }


}


//.map(|&x| if x < 0 { 0 } else { x }); //ReLu for multilayer

/*

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory





env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
*/
