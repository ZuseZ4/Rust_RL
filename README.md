# Rust_RL

This is a test repository to learn about Rust, Neural Networks and Reinforcement Learning.
The Neural Network implementations have some real basic optimizations and the forward pass supports parallelization.
However, it comes with some design-flawes and it is significantly limited by not supporting GPUs and not supporting any kind of autodiff. 

# FUNCTIONALITY
1) The following layers have been implemented: Dense, Dropout, Flatten, Reshape  
2) Convolution_Layer (weight updates work, but don't stack them)  
3) The following activation functions have been implemented: Softmax, Sigmoid, ReLu, LeakyReLu  
4) The following loss functions have been implemented: MSE, RMSE, binary_crossentropy, categorical_crossentropy  
5) The following optimizer have been implemented: SGD (default), Momentum, AdaGrad, RMSProp, Adam  
6) Networks work for 1d, 2d, or 3d input. Exact input shape has to be given, following layers adjust accordingly.  
7) Available Datasets: Mnist(-Fashion), Cifar10, Cifar100
8) Available Agents: Random, Q-Learning, DQN, Double-DQN
9) Available Environments: TicTacToe, Fortress (https://www.c64-wiki.com/wiki/Fortress_(SSI)

# EXAMPLES
MNIST (achieving ~98%)
CIFAR10 (achieving ~49%)

TicTacToe (results in an optimal Agent)
Fortress (... at least DDQN performs significantly better than a random moving bot)


# TODO
1) Add backpropagation of error to conv_layers  
2) Improve design
3) Add GPU support for matrix-matrix multiplications
4) Add some autodiff support.

At least the last two TODOs will probably stay for some time.
