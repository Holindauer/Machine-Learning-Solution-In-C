#include "libraries.h"
#include "structs.h"
#include "functions.h"

void init_model(Model* model) {

	// set shapes 
	model->W_1_rows = LAYER_1_NEURONS;
	model->W_1_cols = NUM_FEATURES;
	model->W_2_rows = LAYER_2_NEURONS;
	model->W_2_cols = LAYER_1_NEURONS;

	model->b_1_cols = LAYER_1_NEURONS;
	model->b_2_cols = LAYER_2_NEURONS;

	// init layer 1 weights, baises, and gradients
	for (int row = 0; row < LAYER_1_NEURONS; row++) {
		model->b_1[row] = 0;
		model->b_1_Grad[row] = 0;
		for (int col = 0; col < NUM_FEATURES; col++) { 
			model->W_1[INDEX(row, col, model->W_1_cols)] = 0;
			model->W_1_Grad[INDEX(row, col, model->W_1_cols)] = 0;
		}
	}
	// init layer 2 weights, biases, and gradients
	for (int row = 0; row < LAYER_2_NEURONS; row++) { 
		model->b_2[row] = 0;
		model->b_2_Grad[row] = 0;
		for (int col = 0; col < LAYER_1_NEURONS; col++) {
			model->W_2[INDEX(row, col, model->W_2_cols)] = 0;
			model->W_2_Grad[INDEX(row, col, model->W_2_cols)] = 0;
		}
	}

	// init network states throughout the forward pass
	for (int example = 0; example < NUM_EXAMPLES; example++) {

		// init input states
		for (int i = 0; i < NUM_FEATURES; i++) {
			model->input[example][i] = 0;
		}
		// init hidden states
		for (int i = 0; i < LAYER_1_NEURONS; i++) {
			model->input[example][i] = 0;
		}
		// init output states
		for (int i = 0; i < LAYER_2_NEURONS; i++){
			model->input[example][i] = 0;
		}
	}	
}


// This is the model's forward pass function
void forward(Model* model, double example[NUM_FEATURES]) {

	// layer 1
	MatMul(model->hidden, model->W_1, example, model->W_1_rows, model->W_1_cols, NUM_FEATURES); // weights * input
	Elementwise_Addition(model->hidden, model->b_1, LAYER_1_NEURONS, 1);						// add bias
	Elementwise_ReLU(model->hidden, LAYER_1_NEURONS, 1);			                         	// ReLU activation
	
	// layer 2
	MatMul(model->output, model->W_2, model->hidden, model->W_2_rows, model->W_2_cols, LAYER_1_NEURONS); // weights * input
	Elementwise_Addition(model->output, model->b_2, LAYER_2_NEURONS, 1);                                 // add bias



	// prediction


}

/*
	The softmax function normalizes the input vector into a probability distribution, where 
	each element of the output vector represents the probability of the corresponding class,
	and the sum of all these probabilities is 1.
 
	for a vector x = [x_1, x_2, ...., x_n]
	softmax(x)_i = exp(x_i) / n_sigma_j=1 [exp(x_j)]
*/
void Elementwise_Softmax(double* matrix, int vector_length) {

	// compute denominator of softmax(x)_i
	double sum = 0;
	for (int i = 0; i < vector_length; i++) {
		sum += exp(matrix[i]);
	}

	// compute softmax over terms


}