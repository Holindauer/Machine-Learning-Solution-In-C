#include "functions.h"
#include "structs.h"
#include "libraries.h"

/*
	Game Plan for Implementing Backpropagation into the model:
	--------------------------------------------------------------------------------

	for epoch in range epochs:
		
		for batch in range(num_batches):   <----- this will need to be adjusted for the specific C implementation

			run batch



*/


int main(void)
{

	//---------------------------------------------------------------------------------------------------------------------Create Network

	// create instances of network weights
	weights net;


	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Layer 1
	
	net.W_1 = malloc(W_1_ELEMENTS * sizeof(double));   // allocate heap memory for weight matrix 1
	check_memory_allocation(net.W_1);

	// set shape of weight matrix 1
	net.W_1_rows = LAYER_1_NEURONS;                    //     W_1    @  Input   = hidden
	net.W_1_cols = INPUT_FEATURES;                     // [128, 784] @ [784, 1] = [128, 1]

	net.b_1 = malloc(128 * sizeof(double));            // bias matrix has same shape as hidden 
	check_memory_allocation(net.b_1);

	net.b_1_rows = 128, net.b_1_cols = 1;


	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Layer 2

	net.W_2 = malloc(W_2_ELEMENTS * sizeof(double));   // allocate heap memory for weight matrix 2
	check_memory_allocation(net.W_2);
	
	// set shape of weight matrix 2
	net.W_2_rows = LAYER_2_NEURONS;                    //   W_2     @  hidden  = model_output
	net.W_2_cols = LAYER_1_NEURONS;                    // [10, 128] @ [128, 1] = [10, 1]

	net.b_2 = malloc(10 * sizeof(double));
	check_memory_allocation(net.b_2);

	net.b_2_rows = 10, net.b_2_cols = 1;               // bias matrix has same shape as hidden 


	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - initialize weights and biases using he intitialization

	he_initialize(net.W_1, net.W_1_rows, net.W_1_cols);     // weights
	he_initialize(net.W_2, net.W_2_rows, net.W_2_cols);
	he_initialize(net.b_1, net.b_1_rows, net.b_1_cols);     // biases
	he_initialize(net.b_2, net.b_2_rows, net.b_2_cols);


	//---------------------------------------------------------------------------------------------------------------------Load in Data

	// set file name
	const char* filename = "mnist_test.csv";

	// create array of examples for dataset. with 100 examples
	example dataset[100];
	initialize_dataset(dataset);

	load_data(filename, dataset, 100);
	

	//---------------------------------------------------------------------------------------------------------------------Run Batch and Compute Loss

	/*
		Currently, batching is done by adjusting the index we are pulling examples
		from the dataset, by adjusting the index at each epoch (this will be more 
		applicable when the training loop is developed). Model outputs for the 
		batch are stored in the batch_outputs struct array.
	*/

	
	batch_outputs batch[8];

	int batch_size = 8;

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









	//---------------------------------------------------------------------------------------------------------------------Free Memory When Done
	// free weight matricies when done
	free(net.W_1);
	free(net.W_2);
	free(net.b_1);
	free(net.b_2);
	
	// free dataset when done
	free_dataset(dataset, 100);
	

	return 0;
}