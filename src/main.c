#include "functions.h"
#include "structs.h"
#include "libraries.h"


int main(void)
{
	printf("\n\nProgram Has Started...");

	//---------------------------------------------------------------------------------------------------------------------Initialize Neural Network

	network net;

	net.batch_size = 8;
	init_model(&net);

	printf("\n\nNetwork has been initialized...");

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - initialize weights and biases using he intitialization

	he_initialize(net.W_1, net.W_1_rows, net.W_1_cols);    
	he_initialize(net.W_2, net.W_2_rows, net.W_2_cols);
	he_initialize(net.b_1, net.b_1_rows, net.b_1_cols);     
	he_initialize(net.b_2, net.b_2_rows, net.b_2_cols);

	printf("\n\nWeights have been randomly initialized w/ he initialization...");

	//---------------------------------------------------------------------------------------------------------------------Load in Data

	const char* filename = "mnist_test.csv";

	// create array of example structs 
	example train_dataset[1000];
	int num_examples = 1000;

	initialize_dataset(train_dataset, 1000);

	load_data(filename, train_dataset, 1000);

	printf("\n\nTrain dataset has been loaded in...");
	

	//---------------------------------------------------------------------------------------------------------------------Train Model
	
	// init training variables
	int epochs = 1000 / 8,  batch_start_idx = 0;

	double loss = 0, learning_rate = 0.001;

	outputs batch[8];    // <--- contains output vector arrray and target

	init_batches(batch, net.batch_size);
	

	// Training Loop
	for (int epoch = 0; epoch < epochs; epoch++) 
	{

		// Compute forward pass for each batch
		for (int b = 0; b < net.batch_size; b++)
		{
			// set target for example
			batch[b].target = train_dataset[b].label;    

			// copy input vector for current example into outputs struct
			for (int i = 0; i < 784; i++) {
				batch[b].input_vector[i] = train_dataset[b].image[i];  
			}

			// run forward pass on  single example
			forward(net, train_dataset[b].image, batch[b].output_vector, b);  
		}

		// backward pass
		backprop(&net, batch);

		// update weights
		gradient_descent(&net, learning_rate);

		
		loss = cross_entropy_loss(batch, net.batch_size);

		printf("\n\n Epoch %d ------ Loss: %lf", (epoch+1), loss);

		// update start index of the batch
		batch_start_idx += net.batch_size;
	}

	//---------------------------------------------------------------------------------------------------------------------Free Memory When Done

	free_dataset(train_dataset, 100);
	free_network(&net);
	
	return 0;
}