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
	double prediction = predict(model_output);


	//--------------------------------------------------------------------------Load in Data

	// set file name
	const char* filename = "mnist_test.csv";

	// create array of examples for dataset. with 100 examples
	example dataset[100];
	initialize_dataset(dataset);

	load_data(filename, dataset, 100);
	

	//--------------------------------------------------------------------------Run Batch and Compute Loss

	/*
		Currently, batching is done by adjusting the index we are pulling examples
		from the dataset, by adjusting the index at each epoch (this will be more 
		applicable when the training loop is developed). Model outputs for the 
		batch are stored in the batch_outputs struct array.
	*/

	int batch_size = 8;

	batch_outputs batch[8];

	for (int i = 0; i < batch_size; i++)             // initialize batch output vectors
	{
		for (int j = 0; j < LAYER_2_NEURONS; j++)
		{
			batch[i].output_vector[j] = 0;
		}
	}

	for (int i = 0; i < batch_size; i++)
	{
		batch[i].target = dataset[i].label;   // set label for current example within batch struct

		forward(net, dataset[i].image, batch[i].output_vector);    // run forward pass on batch
	}


	double loss = cross_entropy_loss(batch, batch_size);

	printf("\n\nLoss: %lf\n\n", loss);



	//--------------------------------------------------------------------------Free Memory When Done
	// free weight matricies when done
	free(net.W_1);
	free(net.W_2);
	
	// free dataset when done
	free_dataset(dataset, 100);
	

	return 0;
}