#include "functions.h"
#include "structs.h"
#include "libraries.h"


int main(void)
{

	//--------------------------------------------------------------------------Create Network

	// create instances of network weights
	weights net;

	net.W_1 = malloc(W_1_ELEMENTS * sizeof(double));   // allocate heap memory for weight matrix 1
	check_memory_allocation(net.W_1);

	// set shape of weight matrix 1
	net.W_1_rows = LAYER_1_NEURONS;                    //     W_1    @  Input   = hidden
	net.W_1_cols = INPUT_FEATURES;                     // [128, 784] @ [784, 1] = [128, 1]


	net.W_2 = malloc(W_2_ELEMENTS * sizeof(double));   // allocate heap memory for weight matrix 2
	check_memory_allocation(net.W_2);
	
	// set shape of weight matrix 2
	net.W_2_rows = LAYER_2_NEURONS;                    //   W_2     @  hidden  = model_output
	net.W_2_cols = LAYER_1_NEURONS;                    // [10, 128] @ [128, 1] = [10, 1]


	// initialize weights using he intitialization
	he_initialize(net.W_1, net.W_1_rows, net.W_1_cols);
	he_initialize(net.W_2, net.W_2_rows, net.W_2_cols);

	//--------------------------------------------------------------------------Run Example Forward Pass

	// intitialize model output
	double model_output[10] = { 0 };

	// initialize dummy example
	double dummy_example[784];

	for (int i = 0; i < 784; i++)
	{
		dummy_example[i] = random_double(0, 1);  // populate dummy example with values between 0 and 1
	}

	// run forward pass
	forward(net, dummy_example, model_output);

	// determine prediction from output vector
	int prediction = predict(model_output);

	// display probability vector output by model
	display_matrix(model_output, 10, 1);
	printf("\n\n Prediction: %d\n\n", prediction);  // and pred


	// free weight matricies when done
	free(net.W_1);
	free(net.W_2);


	//--------------------------------------------------------------------------Load in Data

	// set file name
	const char* filename = "mnist_test.csv";

	// create array of examples for dataset. with 100 examples
	example dataset[100];

	for (int i = 0; i < 100; i++)   // initialize all examples of the dataset
	{
		for (int j=0; j < 784; j++)
		{
			dataset[i].image = malloc(784 * sizeof(double));
			check_memory_allocation(dataset[i].image);
		}
	}

	load_data(filename, dataset, 100);

	

	
	free_dataset(dataset, 100);
	

	return 0;
}