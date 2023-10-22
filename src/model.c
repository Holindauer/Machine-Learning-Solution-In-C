#include "functions.h"
#include "structs.h"
#include "libraries.h"

void init_model(weights* net)
{
	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Layer 1

	// Layer 1 Weights
	net->W_1 = malloc(W_1_ELEMENTS * sizeof(double));
	check_memory_allocation(net->W_1);

	// grads of cost w.r.t. W_1
	net->W_1_grad = malloc(W_1_ELEMENTS * sizeof(double));
	check_memory_allocation(net->W_1_grad);
	arr_init_zero(net->W_1_grad, W_1_ELEMENTS);

	net->W_1_rows = LAYER_1_NEURONS;                          //     W_1    @  Input   = hidden
	net->W_1_cols = INPUT_FEATURES;                           // [128, 784] @ [784, 1] = [128, 1]


	// Layer 1 Bias Vector
	net->b_1 = malloc(LAYER_1_NEURONS * sizeof(double));
	check_memory_allocation(net->b_1);

	// Grads of Cost w.r.t. b_1
	net->b_1_grad = malloc(LAYER_1_NEURONS * sizeof(double));
	check_memory_allocation(net->b_1_grad);
	arr_init_zero(net->b_1_grad, LAYER_1_NEURONS);

	net->b_1_rows = LAYER_1_NEURONS, net->b_1_cols = 1;

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - hidden state

	init_hidden(net);

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Layer 2

	// Layer 2 weight Matrix
	net->W_2 = malloc(W_2_ELEMENTS * sizeof(double));
	check_memory_allocation(net->W_2);

	// Grads of cost w.r.t . W_2 
	net->W_2_grad = malloc(W_2_ELEMENTS * sizeof(double));
	check_memory_allocation(net->W_2_grad);
	arr_init_zero(net->W_2_grad, W_2_ELEMENTS);

	// set shape of weight matrix 2
	net->W_2_rows = LAYER_2_NEURONS;       //   W_2     @  hidden  = model_output
	net->W_2_cols = LAYER_1_NEURONS;       // [10, 128] @ [128, 1] = [10, 1]

	// Layer 2 Bias Vector
	net->b_2 = malloc(LAYER_2_NEURONS * sizeof(double));
	check_memory_allocation(net->b_2);

	// Grads of cost w.r.t. b_2
	net->b_2_grad = malloc(LAYER_2_NEURONS * sizeof(double));
	check_memory_allocation(net->b_2_grad);
	arr_init_zero(net->b_2_grad, LAYER_2_NEURONS);

	net->b_2_rows = LAYER_2_NEURONS, net->b_2_cols = 1;
}

void forward(weights net, double* example, double* model_output)
{
	// example dimmension for mnist
	int X_rows = 784, X_cols = 1;

	// create array to store hidden state
	double hidden[128] = { 0 };


	//----------------------------------------------------------------------------layer 1
	
	// multiply weight matrix 1 by examples to get hdiden state
	matmul(hidden, net.W_1, example, net.W_1_rows, net.W_1_cols, X_rows, X_cols);

	// add bias
	add_bias(hidden, net.b_1, net.b_1_rows, net.b_1_cols);

	// apply ReLU to hidden state
	ReLU(hidden, 128, 1);

	//----------------------------------------------------------------------------layer 2

	// multiply hidden state by weight matrix 2 to get model output
	matmul(model_output, net.W_2, hidden, net.W_2_rows, net.W_2_cols, 128, 1);

	// add biase
	add_bias(model_output, net.b_2, net.b_2_rows, net.b_2_cols);

	// apply softmax to model output
	Softmax(model_output, 10);

}

/*
	This function recieves a vector of softmax applied probabilities,
	to which it comptues the argmax in order to find the model prediction

	it is assumed the length of the model_output is 10 for the task of mnist
*/
double predict(double* model_output)
{
	int arg_max = 0;
	double highest_probability = 0;

	for (int i = 0; i < 10; i++)
	{
		if (model_output[i] > highest_probability)  // check for new max prob
		{
			highest_probability = model_output[i];  // if so, set new highest prob
			arg_max = i;                            // and assign i'th index as new argmax
		}
	}
	return arg_max;
}