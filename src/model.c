#include "functions.h"
#include "structs.h"
#include "libraries.h"

void forward(weights net, double* example, double* model_output)
{
	// example dimmension for mnist
	int X_rows = 784, X_cols = 1;

	// create array to store hidden state
	double hidden[128] = { 0 };


	//----------------------------------------------------------------------------layer 1
	
	// multiply weight matrix 1 by examples to get hdiden state
	matmul(hidden, net.W_1, example, net.W_1_rows, net.W_1_cols, X_rows, X_cols);

	// apply ReLU to hidden state
	ReLU(hidden, 128, 1);

	//----------------------------------------------------------------------------layer 2

	// multiply hidden state by weight matrix 2 to get model output
	matmul(model_output, net.W_2, hidden, net.W_2_rows, net.W_2_cols, 128, 1);

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