#include "libraries.h"
#include "structs.h"
#include "functions.h"

void init_model(Model* model) {

	// init layer 1 weights, baises, and gradients
	for (int row = 0; row < LAYER_1_NEURONS; row++) {
		model->b_1[row] = 0;
		model->b_1_Grad[row] = 0;
		for (int col = 0; col < NUM_FEATURES; col++) { 
			model->W_1[row][col] = 0;
			model->W_1_Grad[row][col] = 0;
		}
	}
	// init layer 2 weights, biases, and gradients
	for (int row = 0; row < LAYER_2_NEURONS; row++) { 
		model->b_2[row] = 0;
		model->b_2_Grad[row] = 0;
		for (int col = 0; col < LAYER_1_NEURONS; col++) {
			model->W_2[row][col] = 0;
			model->W_2_Grad[row][col] = 0;
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