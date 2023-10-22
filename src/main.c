#include "functions.h"
#include "structs.h"
#include "libraries.h"


int main(void)
{

	//---------------------------------------------------------------------------------------------------------------------Initialize Neural Network

	weights net;

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

	initialize_dataset(dataset);

	load_data(filename, dataset, 100);
	

	//---------------------------------------------------------------------------------------------------------------------Run Batch and Compute Loss

	
	batch_outputs batch[8];              // batch_output struct contains output vector arrray and target

	init_batches(batch, net.batch_size);

	int batch_element = 0; // this will need to be incremented at each epoch step once the training loop is developed

	// run forward pass across batch
	for (int i = 0; i < net.batch_size; i++)
	{
		batch[i].target = dataset[i].label;                      // set target

		forward(net, dataset[i].image, batch[i].output_vector, batch_element);  // run forward pass on  single example
	}

	// backprop (under development)
	backprop_W_2(&net, batch);
	backprop_b_2(&net, batch);


	double loss = cross_entropy_loss(batch, net.batch_size);

	printf("\n\nLoss: %lf\n\n", loss);


	//---------------------------------------------------------------------------------------------------------------------Free Memory When Done

	free_dataset(dataset, 100);
	free_network(&net);
	
	return 0;
}