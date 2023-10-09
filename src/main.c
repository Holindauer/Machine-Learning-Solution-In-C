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
	float* W_2 = define_new_matrix(layer_1_nodes, layer_2_nodes);    // layer 2 weight matrix
	float* W_3 = define_new_matrix(layer_2_nodes, layer_3_nodes);    // layer 3 weight matrix

	// Randomly initialize weights of matrix using he initialization
	he_initialize(W_1, input_features, layer_1_nodes);
	he_initialize(W_2, layer_1_nodes, layer_2_nodes);
	he_initialize(W_3, layer_2_nodes, layer_3_nodes);

	





	return 0;
}