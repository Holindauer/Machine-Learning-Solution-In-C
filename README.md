# Neural Network Written in C

This repository contains an implementation of reverse mode automatic differentiation for the purpose of training simple multi layer perceptron neural networks for classification. This was implemented purely in the C programming language.


# Autodiff

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from a vector to a scalar. In the context of deep learning, the gradient of a neural network is used to apply the gradient descent algorithm during training. 

Autodiff is a computational implementation of backpropagation, which is the gereral term for computing the gradient of a function by pasing an input throught the function, saving intermediate computations, and computing the partial derivative with respect to each parameter by recursively applying the chain rule. Backprop is need in situations where it is too complicated (or infeasible) to compute an explicit derivative via more traditional differentiation strategies, like neural networks.
