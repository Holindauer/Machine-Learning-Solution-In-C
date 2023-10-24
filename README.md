# Neural Network Written in C

### Project Overview:
The goal of this repository is to create a Neural Network in the C programming language. Why? 
To gain a greater understanding of underlying computer hardware and reinforce my understanding 
of backpropagation and what neural networks are from a linear algebra perspective. 

I am very new to writing C, so I imagine this will be a fairly intensive process to figure out how 
to do this, so expect rapid CI/CD within this repository.

### Mathematical Representation of Model and Training W/ Backprop + SGD
----------------------------------------------------------------
For the Network Defined as:

	hidden = ReLU(W_1*input + b_1)
	output = Softmax(W_2*hidden + b_2)

	Where W_1, W_2, b_1, b_2 are weight and bias vectors.

----------------------------------------------------------------
The forward pass of this netowrk is defined as: 

	z_1 = W_1 * input + b_1    <---- hidden layer pre activations
	hidden = ReLU(z_1)         <---- hidden layer post activation

	z_2 = W_2 * hidden + b_2   <---- output layer pre activations
	output = Softmax(z_2)      <---- output layer post activation

----------------------------------------------------------------
The above model will be optimized using cross entropy for the loss:

	L(y, p) = - k_Sigma_i=1[y_i * log(p_i)]

	Where y_i is 1 if the true class is i and 0 otherwise, 
	and p_i is the predicted likelihood the i'th example belongs to 
	class i.

----------------------------------------------------------------
Backward Pass:

1.) In order to propagate the gradients backwards, we must start at 
the grad of the loss w.r.t. the Softmax outputs and move backwards:

	dL/dp_i = -y_i / p_i

	Where y_i is 1 if the true class is i and 0 otherwise, 
	and p_i is the predicted likelihood the i'th example 
	belongs to class i.


2.) Compute grad of Softmax outputs w.r.t. the Logits for a particular 
class i:
	
	dSoftmax(z_i)/dz_i = Softmax(z_i_ * (delta_ij - Softmax(z_i))

	Where deltaij is the Kronecker delta, which is 1 when i=j
	and 0 otherwise. 

3.) Gradient of the loss w.r.t. logits 
	
	dL/dz_i = Sigma_j [(dL/dp_j) * (dp_j / dz_i)]

	which, if we substitute in the expressions from step 1 and 2, becomes:

	dL/dz_i = p_i - y_i
	
	Puting this in the context of the network defined above:

	dL/dz_2 = output - true_labels  


4.)  Compute Gradient w/ respect to W_2 and b_2


	dL/Wij = (dL/dz_i) * (dz_i/dWij)
	       = (p_i - y_i) [outer product] h_j

	dL/dW_2 = dL/dz_2 [outer product] hidden
	dL/db_2 = dL/dz_2

5.) Propagate the gradfient through the output layer to the hidden layer
	
	dL/dhidden = W_2.Transpose * dL/dz_2

6.) Compute the gradient w/ respect to z_1 (considering ReLU activation)
	
	dL/dz_1 = dL/dhidden [elementwise multiplication] ReLU'(z_1)
	
	where ReLU'(z_1) is the derivative of ReLU for the values of z_1

7.) Compute the gradient w/ respect to W_1 and b_1:

	dL/dW_1 = dL/dz_1 [outer product] input
	
	dL/db_1 = dL/dz_1


----------------------------------------------------------------
After this Stochsastic Gradient Descent can be Applied using the 
gradient descent learning rule over gradients accumulated over the
batch:

	W_1 = W_1 - alhpa(dL/dW_1)
	b_1 = b_1 - alhpa(dL/db_1)

	W_2 = W_2 - alhpa(dL/dW_2)
	b_2 = b_2 - alhpa(dL/db_2)

	Where alpha is the learning rate


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


### Next Steps:
    Currently, I am working to build backpropagation into the program. The implementation I am
    using will be very specific to the model i am training and likely will require substatial 
    modification in order to be used with a different model.

