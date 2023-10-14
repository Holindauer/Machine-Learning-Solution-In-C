#include "functions.h"
#include "structs.h"
#include "libraries.h"


int main(void)
{
	// seed random with time
	srand((unsigned int)time(NULL));

	// ------------------------ load in data ------------------------ 

	const char* filename = "mnist_test.csv";           // specifiy csv file name containing flattened mnist digits

	int num_examples = 10;		                       // specify number of examples to pull from the dataset csv file

	dataset data = load_data(filename, num_examples);  // load data into dataset struct




	// ------------------------ initialize neural network ------------------------ 


	network_weights net;         // declare an instance of network_weights struct

	int input_features = 784;    // set input features

	int layer_1_neurons = 128,   // set num neurons per layer
		layer_2_neurons = 64,
		layer_3_neurons = 10;    // <--- output shape


	// Create Weight Matrixies

	net.W_1 = create_matrix(input_features, layer_1_neurons);    // layer 1 weights
	net.W_1_rows = input_features;
	net.W_1_cols = layer_1_neurons;

	net.W_2 = create_matrix(layer_1_neurons, layer_2_neurons);   // layer 2 weights
	net.W_2_rows = layer_1_neurons;
	net.W_2_cols = layer_2_neurons;

	net.W_3 = create_matrix(layer_2_neurons, layer_3_neurons);   // layer 3 weights
	net.W_3_rows = layer_2_neurons;
	net.W_3_cols = layer_3_neurons;


	// Randomly Initialize Weights with He Initialization 
	he_initialize(net.W_1, net.W_1_rows, net.W_1_cols);    // layer 1
	he_initialize(net.W_2, net.W_2_rows, net.W_2_cols);    // layer 2
	he_initialize(net.W_3, net.W_3_rows, net.W_3_cols);    // layer 3



	// ------------------------ Test a Forward Pass ------------------------ 


	double* model_output = forward_pass(net, data.examples[0]);






	// free memory after done 
	free_data(&data, num_examples);
	free_model_weights(&net);

	return 0;
}
