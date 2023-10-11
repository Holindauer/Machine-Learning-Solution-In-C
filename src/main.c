#include "libraries.h"

int main(void)
{
	// seed randoim with time
	srand((unsigned int)time(NULL));


	// ------------------------ create multi layer perceptron ------------------------ 
	/*
		Below I am instantiating the weight matricies for a multi layer perceptron with 3 layers.

		Weight matricies,have the shape [num_neurons, input_features]. By multiplying WX, where W is the
		weight matrix for a layer, and X is the input matrix with shape [input_features, 1], the product
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


	// initialize input
	int X_rows = 10, X_cols = 1;
	float* X = define_new_matrix(X_rows, X_cols);		// define input matrix

	// run forward pass
	int y_rows = 30, y_cols = 1;						// initialize output shape
	float* y_hat = forward_pass(W_1, W_2, W_3,			// weight matricies
		layer_1_nodes, layer_2_nodes, layer_3_nodes,    // num neurons per layer
		X, X_rows, X_cols);								// input with shape






	// ------------------------ load in mnist data ------------------------ 


	// specifiy csv file containing flattened mnist digits
	const char* filename = "mnist_test.csv";

	// specify number of examples to pull from the dataset csv file
	int num_examples = 100;

	// create dataset --- as a pointer array to flattned pointer array
	//                    matricies (indexable using the INDEX macro)
	dataset mnist_dataset = load_mnist_digits(filename, num_examples);



	// ------------------------ create a batch of data ------------------------ 
	
	// set batch size 
	int batch_size = 8;

	// set start index of the batch
	int start_of_batch = 0;

	// create new batch
	batch new_batch = gather_batch(mnist_dataset, batch_size, start_of_batch);








	// ------------------------ free memory after program completes ------------------------ 

	free(W_1), free(W_2), free(W_3); // neural network weights
	free(X), free(y_hat);			 // input and output matrices

	free(mnist_dataset.examples);    // dataset
	free(mnist_dataset.labels);

	free(new_batch.examples);        // batch
	free(new_batch.labels);

	return 0;
}