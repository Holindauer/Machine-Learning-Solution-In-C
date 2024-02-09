# Neural Network Written in C

This repository contains an implementation of reverse mode automatic differentiation for training of simple multi layer perceptron neural networks for classification. The entire autodiff implementation was implemented purely in the C programming language. 

Also, Thanks to Andrej Karpathy for the inspo for this project. This C implementation is based on [micrograd](https://github.com/karpathy/micrograd) 

# What is Autodiff?

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from $\mathbb{R^n} \rightarrow \mathbb{R}$. In the context of deep learning, the gradient of a neural network is used for the gradient descent algorithm during trainingin order to adjust the weights of neural network towards better performance.

Autodiff programatically computes the gradient of functions too complicated to find an explicit derivative expression of. Autodiff is an implementation of backpropagation, the process of computing the gradient of a function by recursively applying the chain rule backwards from the output state. The mechanism of autodiff is to track a directed acyclic graph of all computation that went into producing some single scalar value. By splitting up a large function into its elementary components (adds, multiplies, ReLUs in the case of a nn) each partial derivative is found by recursively computing the simple partial derivatives of these elementary operations wrt each ancestor in the graph while moving backwards through it.

This repository contains an implemention of this algorithm, and uses it to train a multi layer perceptron neural network of variable size that can be determiend at runtime. 
