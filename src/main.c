#include "libraries.h"

int main(void)
{
	// seed randoim with time
	srand((unsigned int)time(NULL));


	// ------------------------ create multi layer perceptron ------------------------ 
	/*
		Below I am instantiating the weight matricies for a multi layer perceptron with 3 layers.

		Weight matricies,have the shape [num_neurons, input_features]. By multiplying WX, where W is the 
		weight matrix for a layer, and X is the input matrix wiht shape [input_features, 1], the product 
		matrix will have the shape, [num_neurons, 1] and can be passed into the next layer just as X was 
		passed in. 
		
	*/
	
	//define num input features 
	int input_features = 10;

	// initialize num layer neurons
	int layer_1_nodes = 10,
		layer_2_nodes = 20,
		layer_3_nodes = 30;

	// weight matricies are defined using the define_new_matrix() function from matmul.h

	float* W_1 = define_new_matrix(input_features, layer_1_nodes);   // layer 1 weight matrix
			
	int W_1_rows = input_features,								     
		W_1_cols = layer_1_nodes; 

	float* W_2 = define_new_matrix(layer_1_nodes, layer_2_nodes);    // layer 2 weight matrix

	int W_2_rows = layer_1_nodes,
		W_2_cols = layer_2_nodes;


	float* W_3 = define_new_matrix(layer_2_nodes, layer_3_nodes);    // layer 3 weight matrix

	int W_3_rows = layer_2_nodes,
		W_3_cols = layer_3_nodes;


	// Randomly initialize weights of matrix using he initialization
	he_initialize(W_1, input_features, layer_1_nodes);
	he_initialize(W_2, layer_1_nodes, layer_2_nodes);
	he_initialize(W_3, layer_2_nodes, layer_3_nodes);


	// ------------------------ testing of forward pass ------------------------ 

	/*
		The forward pass of this network will involve a chain of matrix multiplications 
		on the intitial input matrix. 

		LAYER 1:  The input feature X will be passed into the first layer by:

		(W_1)X  = Z_1      ----- Where :   Z_1 is the output of the first layer
										   W_1 is the first layer weight matrix
										   X is the input example
										
										   The shapes of this operation are:  
										   [layer_1_nodes, input_features] @ [input_features, 1] = [layer_1_nodes, 1] = Z_1
		
		LAYER 2:           

		(W_2)(Z_1) = Z_2                   With shapes: [layer_2_nodes, layer_1_nodes] @ [layer_1_nodes, 1] = [layer_2_nodes, 1] = Z_2


		LAYER 3: 

		(W_3)(Z_2) = Z_3                   With shapes: [layer_3_nodes, layer_2_nodes] @ [layer_2_nodes, 1] = [layer_3_nodes, 1] = Z_2
	*/



	// to test the forward pass, I will need to create matricies 
	// for the outputs of each layer as well as an input matrix

	float* X = define_new_matrix(10, 1);								// input 
	int X_rows = 10,
		X_cols = 1;

	float* layer_1_output = define_new_matrix(layer_1_nodes, 1);		// layer 1 output
	float* layer_2_output = define_new_matrix(layer_2_nodes, 1);		// layer 2 output
	float* layer_3_output = define_new_matrix(layer_3_nodes, 1);		// layer 3 output

	


	// forward pass: 
	// 
	// Here is reminder of the protype for the matmult function:
	// void matmul(int* C, int* A, int* B, int rows_A, int cols_A, int rows_B, int cols_B);

	matmul(layer_1_output, W_1, X, W_1_rows, W_1_cols, X_rows, X_cols);
	matmul(layer_2_output, W_2, layer_1_output, W_2_rows, W_2_cols, layer_1_nodes, 1);
	matmul(layer_3_output, W_3, layer_2_output, W_3_rows, W_3_cols, layer_2_nodes, 1);






	// free neural network weights when done
	free(W_1);
	free(W_2);
	free(W_3);
	free(X);
	free(layer_1_output);
	free(layer_2_output);
	free(layer_3_output);

	return 0;
}