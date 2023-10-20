# Neural Network Written in C

### Project Overview:
The goal of this repository is to create a Neural Network in the C programming language. Why? 
To gain a greater understanding of underlying computer hardware and reinforce my understanding 
of backpropagation and what neural networks are from a linear algebra perspective. 

I am new to writing C, so I imagine this will be a fairly intensive process to figure out how 
to go about this, so expect rapid CI/CD within this repository.

### Network Specs:
The specific network I am attempting to train is a an mnist classifier. Specifically, a simple 
multi layer perceptron with one hidden state. 

    hidden = ReLU(W_1 @ X)
    prob_y = Softmax(W_2 @ hidden)

    prediction = argmax(prob_y)

    Where weight matricies:
    W_1 is of size [128, 784]
    W_2 is of size [10, 128]

    Input X is of shape [784, 1]

    hidden is of shape [128, 1]

The reason for choosing this specific model architecture is because I know it will work for minst. Training this exact architecture in torch returns a model that is capable of high 90% accuracy. Thus, reducing my task here to just figuring ut how to implement it into C. 

### Current Status:
    Currently, the program is able to read in labels and mnist pixel intensities flattened into the 
    rows of a csv file. 

    The program also contains functionality to run a forward pass on this loaded-in-data using weights
    and biases initialized with he intitialization. 

### Next Steps:
    Refine the software design of how the program computes the forward pass for a batch of examples. Currently, this is done in main, but I plan to move this, potentially, into its own function so to clean up main().

    Implmenent backpropagation into the forward pass of a batch. There are a number of different ways I could do this that stem from different schools of thought that prioritize either memory conspumption, runtime/efficiency, simplicity of implementation, etc. 
    
    Because I am doing this project as more of a learning exploration of neural networks and C, as well as because this is a very isolated environment where there is only one model being trained on a very specific task, my implementation of backpropagation will not be highly modular. Which is true not just how I intend to implement backpropagation in this project, but for the model as well. 

    To compute the gradients I will use this definition of the chain rule for functions with vector inputs and a single scalar output:

    dz/dx_i = Sigma_j (  (dz/dy_j)(dy_j/dx_i)  )


    Where:

    dz/dx_i is the partial derivative of the cost function with respect to the i'th parameter of the network.

    Sigma_j() refers to a summation of the j inner functions of the composite function y = g(x)  z = f(g(x)) = f(y)

    dz/dy_j is the partial derivative of the cost wrt the j'th inner function

    dy_j/dx_i is the partial derivative of the j'th inner function wrt the i'th parameter of the model


    To compute those partial derivatives, precomputed derivatives will be used in the computation of each of those ancestor parameters. This will likely be the least modular part of the program Thus the implementation of backpropogation in this use case will be highly specific to the network architecture I have chosen to train. 