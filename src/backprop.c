#include "functions.h"
#include "structs.h"
#include "libraries.h"



/*
	 This function computes the values output by the derivative of ReLU elementwise
	 across a matrix.

	 ReLU() = f(x) = max(0, x)

	 Which means it's derivative is:

	 f'(x) = 0    when x <= 0
	 f'(x) = 1    when x > 0

	This derivative is computed in the context of the backpropogation, this mean's
	that the input to the derivative function will be the pre activativations of the model. 
	Whether these values are above or below zero will determine each partial derivative. 

	The input to this function is a matrix of pre_activaitons and a matrix to store
	the computed_derivatives. It also requires the number of elements in the pre activation 
	hidden state.

*/

void ReLU_derivative(double* pre_activations, double* computed_derivatives, int num_elements)
{

	for (int i = 0; i < num_elements; )
	{
		if (pre_activations[i] > 0)
		{
			computed_derivatives[i] = 1;   // f'(x) = 1    when x > 0
		}
		else
		{
			computed_derivatives[i] = 0;   // f'(x) = 0    when x <= 0
		}
	}
}



/*
	A Note on the Mathematical Derivation of the Gradient Computation for the Layer Weight Matrix and Bias Vector

    --------------------------------------------
	Softmax Activation: Given a vector of raw scores  z of size k, where k is the number of classes:

	Softmax(z_i) = exp(z_i) / k_Sigma_j=1[ exp(z_j) ]    for i = {1, ..., K}

	--------------------------------------------
	Cross Entropy Loss Function: Since the last output of the model uses softmax, cross entropy loss is used:

	L(y, p) = - k_Sigma_i=1[y_i * log(p_i)]

	Where y_i is 1 if the true class is i and 0 otherwise, and p_i is the predicted probability for class i.

	--------------------------------------------
	Gradient of the loss w.r.t. the Softmax outputs:

	dL/dp_i = -y_i / p_i

	--------------------------------------------
	Gradient of the Softmax Outputs w.r.t. the Logits z_i for a particular class i is: 

	dSoftmax(z_i)/dz_i = Softmax(z_i) * (delta_ij - Softmax(z_i))

	Where deltaij is the Kronecker delta, whihc is 1 when i=j and 0 otherwise

	--------------------------------------------
	Gradient of the Loss w.r.t. Logits (using the defintion of the chain rule for functions w/ vector inputs):

	dL/dz_i = k_Sigma_j=1 [(dL/dp_j) * (dp_j/dz_i)]

	Which, if we substitude the two above expressions in simplifies to:

	dL/dz_i = p_i / y_i

	--------------------------------------------
	Gradient Computation of the weights and biases of the last layer:

	Where W is the last weight matrix, b the biases, and h is the last hidden state.

	dL/dWij = (dL/dz_i) * (dz_i/dW_ij) = (p_i - y_i) * h_j

	dL/dbi = p_i - y_i
*/



/*
	This function accumulates the gradient of the cost with respect to each weight for the 
	weight matrix of the last layer of the model. 

	dL/dWij = (dL/dz_i) * (dz_i/dW_ij)         <--- To understand the derivation of this
	        = (p_i - y_i) * h_j                     look at the note above the function W_2
				                                    and b_2 grad comp functions in backprop.c
				   

	The function takes the follwing arguments:

		- weights* net is a pointer to the weights struct of model

		- batch_outputs* batch is an array of batch_output structs whose members contains the 
		                       following data for each example predicted on within a batch:

		     - the output vector of of the model, containing the likelihood each class is the target
			 - as well as the label for each example within the array
*/
void backprop_W_2(weights* net, batch_outputs* batch)
{
	double gradient = 0;

	for (int b = 0; b < net->batch_size; b++){    // iterate batches
		for (int i = 0; i < net->W_2_rows; i++){  // iterate rows of weight matrix
			for (int j=0; j < net->W_2_cols; j++){  // iterate cols of weight matrix 

				gradient = (batch[b].output_vector[i] - batch[b].target) * net->hidden[b][j];  // compute gradient  -----  dL/dWij = (p_i - y_i) * h_j    

				net->W_2_grad[INDEX(i, j, net->W_2_cols)] += gradient;                         // accumulate gradient for example
			}
		}
	}
}

/*
	This function accumulates the gradient of the cost with respect to each weight for the
	weight bias vector of the last layer of the model.

	dL/dbi = p_i - y_i      <--- To understand the derivation of this
		                         look at the note above the function W_2
								 and b_2 grad comp functions in backprop.c


	The function takes the follwing arguments:

		- weights* net is a pointer to the weights struct of model

		- batch_outputs* batch is an array of batch_output structs whose members contains the
							   following data for each example predicted on within a batch:

			 - the output vector of of the model, containing the likelihood each class is the target
			 - as well as the label for each example within the array
*/
void backprop_b_2(weights* net, batch_outputs* batch)
{
	double gradient = 0;

	for (int b = 0; b < net->batch_size; b++){         // iterate batches
		for (int i = 0; i < net->b_2_rows; i++){       // iterate rows of bias vector
			for (int j = 0; j < net->b_2_cols; j++){   // iterate cols of bias vector 

				gradient = batch[b].output_vector[i] - batch[b].target;    // compute gradient   -----  dL/dbi = p_i - y_i 

				net->b_2_grad[INDEX(i, j, net->b_2_cols)] += gradient;     // accumulate gradient for example
			}
		}
	}
}
