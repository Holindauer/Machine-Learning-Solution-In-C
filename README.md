# Neural Network Written in C

This repository contains an implementation of reverse mode automatic differentiation for training of simple multi layer perceptron neural networks for classification. This entire autodiff and mlp implementation was written purely in the C programming language. 

Also, Thanks to Andrej Karpathy for the inspo for this project. This C implementation is based on [micrograd](https://github.com/karpathy/micrograd) 

# What is Autodiff?

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from $\mathbb{R^n} \rightarrow \mathbb{R}$. In the context of deep learning, the gradient of a neural network is used for the gradient descent algorithm during training in order to adjust the weights of a neural network towards better performance.

Autodiff programatically computes the gradient of functions too complicated to find an explicit derivative expression of. Autodiff gets around this issue by implementing backpropagation, the process of computing the gradient of a function by recursively applying the chain rule backwards from the output state. The mechanism of autodiff is to track a directed acyclic graph of all computation that went into producing some single scalar value. By splitting up a large function into its elementary components (adds, multiplies, ReLUs in the case of a nn) each partial derivative is found by recursively computing the simple partial derivatives of these elementary operations wrt each ancestor in the graph while moving backwards through it. This simplifies the derivative computation by divide and conquer.

This repository implements this algorithm, and uses it to train a multi layer perceptron neural network of variable size that can be determiend at runtime. 

# How are Neural Networks and Autodiff Represented?

The autodiff implementation relies on a struct called Value. A Value struct has the following 4 important members:

- Value.value
- Value.grad
- Value.Backward
- Value.ancestors

The Value.value member of the struct represents a single scalar value within a computation graph. The Value.ancestors member is an array of Value pointers. These are arguments into some computation that produced the Value struct in question. The .grad member is the partial derivative of the function that produced Value.value for that Value's ancestor's Value.value fields. Value.Backward is a pointer to a function that will compute the partial derivative of the computation the created the Value.

Simple operations are grouped into functions that accept Value structs as arguments and produce a Value struct as output. Add() Multiply() ReLU() in our case. As computation builds up, an intricate graph of these Values is defined, to which backpropagation can be applied. 

I represented neural networks here as a linked list of Layer structs. Where each Layer struct contains a Weight and Bias matrix/vector of Value structs. By Passing some array of value structs through the forward pass, which should produce a single output, backprobagation is thus able to be integrated into the context of a neural network.


