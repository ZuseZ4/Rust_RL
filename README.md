# Rust_ML

Small test repository to learn Rust and reinforcement learning.
Fortress is used as arbitrary choosen testgame:
https://www.c64-wiki.com/wiki/Fortress_(SSI)

neural network based code is in src/network.
actual layers are in src/network/concrete_layers, the neural network is build in src/nn.rs

#EXISTING
1) Dense_Layer, flatten_Layer 
2) softmax, sigmoid, 
3) binary_crossentropy, categorical_crossentropy,


#TODO
1) add conv layers (and (leaky) ReLu along the way)
2) write bot for Fortress using Conv layer
4) add CIFAR10 testcase

#DONE
1) fix error forwarding during backprop when using multiple dense layer
2) extract AND, NOT, OR, First, XOR into own test cases
3) add MNIST example
4) refactoring to work with 2d/3d input
5) Add more examples under /src/examples (based on testcases)
