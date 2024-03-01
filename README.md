# Neural Network Written in C

This repository contains an implementation of reverse mode automatic differentiation for training of simple multi layer perceptron neural networks for classification. This entire autodiff and mlp implementation was written purely ine C. 

Also, Thanks to Andrej Karpathy for the inspo for this project. This C implementation is based on [micrograd](https://github.com/karpathy/micrograd) 

# What is Autodiff?

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from $\mathbb{R^n} \rightarrow \mathbb{R}$. In the context of deep learning, the gradient of a neural network is integral to the gradient descent algorithm during which adjusts the weights of a neural network towards better performance.

Autodiff programatically computes the gradient of functions too complicated to find an explicit derivative expression of. Autodiff gets around this issue by implementing backpropagation, the process of computing the gradient of a function by recursively applying the chain rule backwards from the output state. Autodiff involves tracking a directed acyclic graph of all computation that went into producing some singular scalar value. By splitting up a large function into its elementary components (adds, multiplies, ReLUs...) each partial derivative is determined by recursively computing these simple partial derivatives of elementary operations at each node in the graph wrt its imediate ancestors. This process starts with at the final output and moves backwards. This simplifies the gradient computation via divide and conquer.

This repository implements this algorithm, and uses it to train a multi layer perceptron neural network of which its size can be determiend at runtime. 

# How are Neural Networks and Autodiff Represented?

The autodiff implementation relies on a struct called Value. A Value struct has the following 4 important members:

- Value.value
- Value.grad
- Value.Backward
- Value.ancestors

The Value.value member of the struct represents a single scalar value within a computation graph. The Value.ancestors member is an array of Value pointers. These are arguments into some computation that produced the Value struct in question. The .grad member is the partial derivative of the function that produced Value.value for that Value's ancestor's Value.value fields. Value.Backward is a pointer to a function that will compute the partial derivative of the computation the created the Value.

Simple operations are grouped into functions that accept Value structs as arguments and produce a Value struct as output. Add() Multiply() ReLU() in our case. As computation builds up, an intricate graph of these Values is defined, to which backpropagation can be applied. 







I represented neural networks here as a linked list of Layer structs. Where each Layer struct contains a Weight and Bias matrix/vector of Value structs. By Passing some array of value structs through the forward pass, which should produce a single output, backprobagation is thus able to be integrated into the context of a neural network.


