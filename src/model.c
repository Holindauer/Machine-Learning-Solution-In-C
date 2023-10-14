#include "functions.h"
#include "structs.h"
#include "libraries.h"


/*
	This function performs a forward pass using
	the weights within a network_weights struct
*/
double* forward_pass(network_weights net, double* input_example)
{
	// allocate memory for hidden layer outputs
	double* hidden_1 = malloc(net.W_1_cols * sizeof(double));         // hidden layer 1 output
	double* hidden_2 = malloc(net.W_2_cols * sizeof(double));         // hidden layer 2 output
	double* output_vector = malloc(net.W_3_cols * sizeof(double));    // model output vector

	if (hidden_1 == NULL || hidden_2 == NULL || output_vector == NULL) // error handling 
	{
		free(hidden_1); 
		free(hidden_2);
		free(output_vector);

		exit(1);
	}

	// This input shape is specific to the mnist dataset
	int input_rows = 784, input_cols = 1;


	// run forward pass: 

	matmul(hidden_1, net.W_1, input_example, net.W_1_rows, net.W_1_cols, input_rows, input_cols);   // multiply input by weight matrix
	ReLU(hidden_1, net.W_1_rows, 1);	                                                               // apply activation

	matmul(hidden_2, net.W_2, hidden_1, net.W_2_rows, net.W_2_cols, net.W_1_cols, 1);              // multiply layer 1 output by weight matrix
	ReLU(hidden_2, net.W_2_rows, 1);					                                               // apply activation

	matmul(output_vector, net.W_3, net.W_2, net.W_3_rows, net.W_3_cols, net.W_2_cols, 1);         // multiply layer 2 output by weight matrix
	Softmax(output_vector, net.W_3_cols, 1);                                                          // apply softmax activation for turning
	                                                                                                   // logits -----> probabilities

	return output_vector;
}




/*
	This function frees the weight matricies of an mlp when done
*/
void free_model_weights(network_weights* network)
{
	free(network->W_1);
	free(network->W_2);
	free(network->W_3);
}