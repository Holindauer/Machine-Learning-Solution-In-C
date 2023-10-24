#include "functions.h"
#include "structs.h"
#include "libraries.h"


// NOTE: Fix backpropW_2 and backpropb_2 to use outer product instead of elementise multiplication.



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
	        = (p_i - y_i) [outer propduct] h_j      look at the ReadMe

	The function takes the follwing arguments:

		- weights* net is a pointer to the weights struct of model

		- batch_outputs* batch is an array of batch_output structs whose members contains the 
		                       following data for each example predicted on within a batch:

		     - the output vector of of the model, containing the likelihood each class is the target
			 - as well as the label for each example within the array
*/
void backprop_W_2(weights* net, batch_outputs* batch)
{
	/*
		----- A Note on the Shape of the below Matrix Operations: ------

		This funciton accumulates the gradient of the cost w.r.t. the weights of the
		last weight matrix of the model. However, this is done for each model output
		of the batch. Each gradient is computes then averaged over the batch length.


		Below, the following operation is performed:

		compute dL/dW_2 = dL/dz_2 [outer product] hidden

		Where:   W_2 has shape [10, 128]
				 net->hidden[batch example] has shape [128, 1]

				  Therefore the output product of dL/dz_2 by each
				  hidden state will be: [10, 128]

				  Which is the same shape as W_2. In the below algorithm,
				  each gradient is accumulated into the double* outer_prod
				  array we have allocated memory for. Each outer product is
				  accumulated into net->W_2_grad. After all gradients have
				  been accuumulated for the batch, each partial derivative
				  is averaged across the batch, to which it is readyu to use
				  for stochastic gradient descent.
	*/


	// allocate memory to store arrays for dL/dz_2 for each precomuted forward pass for the batch
	double* dL_dz_2 = malloc(LAYER_2_NEURONS * sizeof(double));
	check_memory_allocation(dL_dz_2);
	arr_init_zero(dL_dz_2, LAYER_2_NEURONS);
	
	// allocate memory for a matrix to hold dL/dz_2 [outer product] hidden
	double* outer_prod = malloc(LAYER_2_NEURONS * LAYER_1_NEURONS * sizeof(double));
	check_memory_allocation(outer_product);                                         
	arr_init_zero(outer_prod, W_2_ELEMENTS);


	for (int b = 0; b < net->batch_size; b++) // iterate through batches
	{
		
		for (int i = 0; i < LAYER_2_NEURONS; i++)            // <--- iterate through output vector size 
		{
			dL_dz_2[i] = batch[b].output_vector[i] - batch[b].target;  // dLdz_2 = output - true_labels
		}

		outer_product(dL_dz_2, net->hidden, outer_prod, LAYER_2_NEURONS, LAYER_1_NEURONS);  // compute outer product of dL/dz_2 and the hidden 
		                                                                                    // state for the current example of the batch

		for (int row = 0; row < LAYER_2_NEURONS; row++) {     // Iterate through the shape of W_2 (which is the same as outer_product)
			for (int col = 0; col < LAYER_1_NEURONS; col++) {

				net->W_2_grad[INDEX(row, col, LAYER_1_NEURONS)] += outer_prod[INDEX(row, col, LAYER_1_NEURONS)]; // accumulate gradient 
			}
		}
	}

	for (int row = 0; row < LAYER_2_NEURONS; row++) {
		for (int col = 0; col < LAYER_1_NEURONS; col++) {
			net->W_2_grad[INDEX(row, col, LAYER_1_NEURONS)] /= net->batch_size;  // average gradients across batch size
		}
	}

	free(outer_prod);
	free(dL_dz_2);     
}

/*
	This function accumulates the gradient of the cost with respect to each weight for the
	weight bias vector of the last layer of the model.

	dL/dbi = p_i - y_i      <--- To understand the derivation of this
		                         look at the ReadMe

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

	for (int i = 0; i < net->b_2_rows; i++) {
		for (int j = 0; j < net->b_2_cols; j++) {
			net->b_2_grad[INDEX(i, j, net->b_2_cols)] /= net->batch_size;  // average gradient across batch
		}
	}
}