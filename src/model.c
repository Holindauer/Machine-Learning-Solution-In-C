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
		model->b_1[row] = rand_double();
		model->b_1_Grad[row] = 69;
		for (int col = 0; col < NUM_FEATURES; col++) { 
			model->W_1[INDEX(row, col, model->W_1_cols)] = rand_double();
			model->W_1_Grad[INDEX(row, col, model->W_1_cols)] = 69;
		}
	}
	// init layer 2 weights, biases, and gradients
	for (int row = 0; row < LAYER_2_NEURONS; row++) { 
		model->b_2[row] = rand_double();
		model->b_2_Grad[row] = 69;
		for (int col = 0; col < LAYER_1_NEURONS; col++) {
			model->W_2[INDEX(row, col, model->W_2_cols)] = rand_double();
			model->W_2_Grad[INDEX(row, col, model->W_2_cols)] = 69;
		}
	}

	// init network states throughout the forward pass
	for (int example = 0; example < NUM_EXAMPLES; example++) {
		// init input states
		for (int i = 0; i < NUM_FEATURES; i++) {
			model->input[example][i] = 69;
		}
		// init hidden states
		for (int i = 0; i < LAYER_1_NEURONS; i++) {
			model->hidden[example][i] = 69;
			model->hidden_Grad[i] = 69;
		}
		// init logit/output states
		for (int i = 0; i < LAYER_2_NEURONS; i++){
			model->logits[example][i] = 69;
			model->logits_Grad[i] = 69;
			model->output[example][i] = 69;
			model->output_Grad[i] = 69;

		}
		// init predictions
		model->prediction[example] = 69;
	}	
}


// This is the model's forward pass function
double forward(Model* model, double example[NUM_FEATURES], int example_num) {

	// save input state
	for (int i = 0; i < NUM_FEATURES; i++) {
		model->input[example_num][i] = example[i];
	}

	// layer 1
	MatMul(model->hidden[example_num], model->W_1, example, model->W_1_rows, model->W_1_cols, NUM_FEATURES); // weights * input
	Elementwise_Addition(model->hidden[example_num], model->b_1, LAYER_1_NEURONS, 1);						// add bias
	Elementwise_ReLU(model->hidden[example_num], LAYER_1_NEURONS, 1);			                         	// ReLU activation
	
	// layer 2
	MatMul(model->output[example_num], model->W_2, model->hidden[example_num], model->W_2_rows, model->W_2_cols, LAYER_1_NEURONS); // weights * input
	Elementwise_Addition(model->output[example_num], model->b_2, LAYER_2_NEURONS, 1);                                 // add bias
	Elementwise_Softmax(model->output[example_num], LAYER_2_NEURONS);                                                 // Softmax activation

	// take argmax of logits for prediction
	double max = 0, max_idx = 0;
	for (int i = 0; i < NUM_CLASSES; i++) {
		if (model->output[example_num][i] > max) {
			max = model->output[example_num][i];
			max_idx = i;
		}
	}
		
	return max_idx;

}



