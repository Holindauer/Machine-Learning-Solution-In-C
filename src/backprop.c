#include "functions.h"
#include "structs.h"
#include "libraries.h"

/*
	 This function computes the derivative of ReLU for a given input to the ReLU() func

	 ReLU() = f(x) = max(0, x)

	 f'(x) = 0    when x <= 0   ---  f'(x) = 1    when x > 0
*/
void ReLU_derivative(double* pre_activations, double* computed_derivatives, int num_elements)
{
	for (int i = 0; i < num_elements; )
	{
		if (pre_activations[i] > 0){ computed_derivatives[i] = 1;   /* f'(x) = 1    when x > 0*/}
		else{ computed_derivatives[i] = 0;                         /*f'(x) = 0    when x <= 0*/}
	}
}

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

	for (int b = 0; b < net->batch_size; b++){      // iterate batches
		for (int i = 0; i < net->W_2_rows; i++){    // iterate rows of weight matrix
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