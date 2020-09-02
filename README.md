# Rust_ML

Small test repository to learn Rust and reinforcement learning.  
Fortress is used as an arbitrary choosen testgame:  
https://www.c64-wiki.com/wiki/Fortress_(SSI)

Neural network based code is in src/network.  
Actual layers are in src/network/concrete_layers.  
The neural network is build in src/nn.rs  

#EXISTING
1) The following layers have been implemented: Dense, Dropout, Flatten, Reshape  
2) Convolution_Layer (weight updates work, but don't stack them)  
3) The following activation functions have been implemented: Softmax, Sigmoid, ReLu, LeakyReLu  
4) binary_crossentropy, categorical_crossentropy  
5) Networks work for 1d, 2d, or 3d input. Exact input shape has to be given, following layers adjust accordingly.
6) Cifar10 & MNIST testcase (achieving ~49% and ~98%) 
7) Q-Learning example (Fortress), Deep-Q-Learning example (Fortress)  


#TODO
1) add backpropagation of error to conv_layers   
2) Finish MSE implementation   

