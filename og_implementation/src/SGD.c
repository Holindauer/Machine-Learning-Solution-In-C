#include "functions.h"
#include "structs.h"
#include "libraries.h"

/*
	This function performs the gradient descent learning rule after
	the gradients of a batch have been computed.

	The gradients and weights are all held inside of the network struct,
	which is passed in as an argument. The other parameter needed is the
	learning rate, which is to be set before the training loop begins 
	within main().

	The gradient descent learning rule is defined as:

	the update for a parameter theta_i = theta_i - ( learning_rate * dL/theta_i)
*/
void gradient_descent(network* net, double learning_rate)
{
	// Update W_2
	for (int row = 0; row < net->W_2_rows; row++) {
		for (int col = 0; col < net->W_2_cols; col++) {

			// theta_i = theta_i - ( learning_rate * dL/theta_i)
			net->W_2[INDEX(row, col, net->W_2_cols)] -= net->W_2_grad[INDEX(row, col, net->W_2_cols)] * learning_rate;  
		}
	}

	// Update b_2
	for (int i = 0; i < net->b_2_rows; i++) {

		// theta_i = theta_i - (learning_rate * dL / theta_i)
		net->b_2[i] -= net->b_2_grad[i] * learning_rate;
	}

	// Update W_1
	for (int row = 0; row < net->W_1_rows; row++) {
		for (int col = 0; col < net->W_1_cols; col++) {

			// theta_i = theta_i - ( learning_rate * dL/theta_i)
			net->W_1[INDEX(row, col, net->W_1_cols)] -= net->W_1_grad[INDEX(row, col, net->W_1_cols)] * learning_rate;
		}
	}

	// Update b_1
	for (int i = 0; i < net->b_1_rows; i++) {

		// theta_i = theta_i - (learning_rate * dL / theta_i)
		net->b_1[i] -= net->b_1_grad[i] * learning_rate;
	}
}


