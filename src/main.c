#include "functions.h"
#include "structs.h"
#include "libraries.h"


int main(void)
{

	//---------------------------------------------------------------------------------------------------------------------Initialize Neural Network

	network net;

	net.batch_size = 8;
	init_model(&net);

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - initialize weights and biases using he intitialization

	he_initialize(net.W_1, net.W_1_rows, net.W_1_cols);    
	he_initialize(net.W_2, net.W_2_rows, net.W_2_cols);
	he_initialize(net.b_1, net.b_1_rows, net.b_1_cols);     
	he_initialize(net.b_2, net.b_2_rows, net.b_2_cols);


	//---------------------------------------------------------------------------------------------------------------------Load in Data

	const char* filename = "mnist_test.csv";

	// create array of example structs 
	example dataset[100];
	int num_examples = 100;

	initialize_dataset(dataset, 100);

	load_data(filename, dataset, 100);
	

	//---------------------------------------------------------------------------------------------------------------------Run Batch and Compute Loss

	
	outputs batch[8];              // batch_output struct contains output vector arrray and target

	init_batches(batch, net.batch_size);


	// run forward pass across batch
	for (int b = 0; b < net.batch_size; b++)
	{
		batch[b].target = dataset[b].label;    // set target

		// copy input vector for current example into outputs struct
		for (int i = 0; i < 784; i++) {
			batch[b].input_vector[i] = dataset[b].image[i];  // <---- memory error here
		}
		

		forward(net, dataset[b].image, batch[b].output_vector, b);  // run forward pass on  single example
	}

	// backward pass
	backprop(&net, batch);
	




	double loss = cross_entropy_loss(batch, net.batch_size);

	printf("\n\nLoss: %lf\n\n", loss);


	//---------------------------------------------------------------------------------------------------------------------Free Memory When Done

	free_dataset(dataset, 100);
	free_network(&net);
	
	return 0;
}