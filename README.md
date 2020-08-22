# Rust_ML

Small test repository to learn Rust and reinforcement learning.  
Fortress is used as arbitrary choosen testgame:  
https://www.c64-wiki.com/wiki/Fortress_(SSI)

neural network based code is in src/network.  
Actual layers are in src/network/concrete_layers.  
The neural network is build in src/nn.rs  

#EXISTING
1) Dense_Layer, flatten_Layer, reshape_Layer
2) Softmax, Sigmoid, ReLu, LeakyReLu
3) binary_crossentropy, categorical_crossentropy,
4) 1d, 2d, or 3d input
5) Cifar10 & MNIST testcase (achieving ~45% and ~96%)
6) Q-Learning example (Fortress)


#TODO
1) add conv layer
2) Add Deep-Q-Learning example
