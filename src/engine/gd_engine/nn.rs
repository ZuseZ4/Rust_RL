use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::normal::StandardNormal;


pub struct NeuralNetwork {
  H: usize,
  batch_size: usize,
  learning_rate = f32,
  gamma = f32,
  decay_rate = f32,
  resume = bool,
  render = bool,
}


impl NeuralNetwork {
  pub fn new(input_dim: usize) -> Neuralnetwork {
    //xavier init
    let NN = Array::random((36,36), StandardNormal::new())
      .map(|&x| x / sqrt(36));

    NeuralNetwork{
      H: 36,//200,
      D: input_dim,
      batch_size = 10,
      learning_rate = 1e-4,
      gamma = 0.99,
      decay_rate = 0.99,
      resume = false,
      render = false,
    }

  pub fn init() {
  }
}

fn softmax(x: Vec<f32>) -> Vec<f32> {
  max = 
}

fn policy_forward(x) -> Vec<f32> {
  let output = W1.dot(x);
    
    //.map(|&x| 1 / (1 + (-x).exp())) //sigmoid
    //.map(|&x| if x < 0 { 0 } else { x }); //ReLu for multilayer
}

fn discount_rewards(moves_prob, legal_moves: Vec<i32>) -> f32 {
  let n = legal_moves.iter().sum() as f32;
  let normalized_legal_moves: Vec<f32> = legal_moves.iter().map(|&x| (x as f32)/num_legal_moves).collect();
  //MSE pushed the neural net into giving all allowed moves the same prob and 0 to the rest
  //+1 for mimicing legal moves perfectly, neg values if it is too far
  let MSE = moves_prob.iter().
    zip(normalized_legal_moves.iter())
    .fold(0.0, |sum, (&x, &y)| sum + ((x-y)*3.0).powf(2.0));
  1 - MSE
}

fn policy_backward(

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

