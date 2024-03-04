# Autodiff and Neural Net Implementation in C

This repository contains an implementation of reverse mode automatic differentiation for the training of simple multi layer perceptron neural networks for the task of classification. This entire implementation was written purely in C. 

Also, Thanks to Andrej Karpathy for the inspo for this project. This C implementation is based on [micrograd](https://github.com/karpathy/micrograd) 

# What is Autodiff?

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from $\mathbb{R^n} \rightarrow \mathbb{R}$. In the context of deep learning, computing the gradient of the loss of a neural network is required for the gradient descent algorithm, which is the algorithm that adjusts the weights of a neural network towards better performance.

Autodiff is a protocol that programatically computes the gradient of functions too complicated to find an explicit derivative expression of. Autodiff gets around this issue by implementing backpropagation, the process of computing the gradient of a function by recursively applying the calculus chain rule backwards from the output state. Backprop is done programatically in autodiff by tracking a directed acyclic graph of all state during the computation that went into producing some single scalar value. By splitting up a large function into its elementary components (adds, multiplies, ReLUs...), of which their partial derivatives are very simple, each partial derivative of an extremely complicated function is constructed backwards from chaining together these simple components' derivatives wrt their imediate ancestors. This process starts with at the final output and moves backwards until all parameters are reached.

This repository implements this algorithm, and uses it to train a multi layer perceptron neural network of variable size at runtime. 

# How is Autodiff Represented in this Repository?

The autodiff implementation is built around the Value struct. A Value has the following 4 important members (there are more members than this in the code, but I have included only the most important ones here for understanding autodiff):

    typedef struct _value {
        double value;             
        double grad;              
        pBackwardFunc Backward; 
        Value** ancestors;
    } Value;

A Value contains a *value*, this is a state at some point in the computational graph. It also has the *grad* member, this represents the partial derivative of the elementary operation that went into creating this particular Value's *value*. The *ancestors* member is an array of Value struct pointers containing the immediate ancestors that went into creating the Value. *pBackwardFunc* is a function pointer that will compute the partial derivative of the particular partial derivative that created this Value wrt to its ancestors. 

Ancestors structs are linked together in a directed acyclic graph.