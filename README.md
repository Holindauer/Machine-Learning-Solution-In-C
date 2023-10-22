# Neural Network Written in C

### Project Overview:
The goal of this repository is to create a Neural Network in the C programming language. Why? 
To gain a greater understanding of underlying computer hardware and reinforce my understanding 
of backpropagation and what neural networks are from a linear algebra perspective. 

I am very new to writing C, so I imagine this will be a fairly intensive process to figure out how 
to do this, so expect rapid CI/CD within this repository.

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

The reason for choosing this specific model architecture is because I know it will work for minst. 
Training this exact architecture in torch returns a model that is capable of high 90% accuracy. Thus, 
reducing my task here to just figuring ut how to implement it into C. 

### Current Status:
    Currently, the program is able to read in labels and mnist pixel intensities flattened into the 
    rows of a csv file. 

    The program also contains functionality to run a forward pass on this loaded-in-data using weights
    and biases initialized with he intitialization. 
    
### A Note on the Derivation of Gradients for the Backprop Computation:
    ----------------------------------------------------------------------------------------
	Softmax Activation: Given a vector of raw scores z of size k, where k is the number of classes:
	Softmax(z_i) = exp(z_i) / k_Sigma_j=1[ exp(z_j) ]    for i = {1, ..., K}
	----------------------------------------------------------------------------------------
	Cross Entropy Loss Function:

	L(y, p) = - k_Sigma_i=1[y_i * log(p_i)]

	Where y_i is 1 if the true class is i and 0 otherwise, and p_i is the predicted probability for class i.
	----------------------------------------------------------------------------------------
	Gradient of the loss w.r.t. the Softmax outputs:

	dL/dp_i = -y_i / p_i
	----------------------------------------------------------------------------------------
	Gradient of the Softmax Outputs w.r.t. the Logits z_i for a particular class i is: 

	dSoftmax(z_i)/dz_i = Softmax(z_i) * (delta_ij - Softmax(z_i))

	Where deltaij is the Kronecker delta, whihc is 1 when i=j and 0 otherwise
	----------------------------------------------------------------------------------------
	Gradient of the Loss w.r.t. Logits (using the defintion of the chain rule for functions w/ vector inputs):

	dL/dz_i = k_Sigma_j=1 [(dL/dp_j) * (dp_j/dz_i)]

	Which, if we substitude the two above expressions in simplifies to:

	dL/dz_i = p_i / y_i
	----------------------------------------------------------------------------------------
	Gradient Computation of the weights and biases of the last layer:

	dL/dWij = (dL/dz_i) * (dz_i/dW_ij) 
            = (p_i - y_i) * h_j

	dL/dbi = p_i - y_i

    Where W is the last weight matrix, b the biases, and h is the last hidden state.

    
### Next Steps:
    Currently, I am working to build backpropagation into the program. The implementation I am
    using will be very specific to the model i am training and likely will require substatial 
    modification in order to be used with a different model.

