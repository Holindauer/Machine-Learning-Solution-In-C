#include "libraries.h"
#include "structs.h"
#include "functions.h"

// This file contains functions related to stochastic gradient descent

void zero_init_gradient_accumulator(Gradient* gradient_accumulator) {

	// set shapes 
	gradient_accumulator->W_1_rows = LAYER_1_NEURONS;
	gradient_accumulator->W_1_cols = NUM_FEATURES;
	gradient_accumulator->W_2_rows = LAYER_2_NEURONS;
	gradient_accumulator->W_2_cols = LAYER_1_NEURONS;

	gradient_accumulator->b_1_cols = LAYER_1_NEURONS;
	gradient_accumulator->b_2_cols = LAYER_2_NEURONS;

	// init layer 1 gradients
	for (int row = 0; row < LAYER_1_NEURONS; row++) {
		gradient_accumulator->b_1_Grad[row] = 69;
		for (int col = 0; col < NUM_FEATURES; col++) {
			gradient_accumulator->W_1_Grad[INDEX(row, col, gradient_accumulator->W_1_cols)] = 69;
		}
	}
	// init layer 2 gradients
	for (int row = 0; row < LAYER_2_NEURONS; row++) {
		gradient_accumulator->b_2_Grad[row] = 69;
		for (int col = 0; col < LAYER_1_NEURONS; col++) {
			gradient_accumulator->W_2_Grad[INDEX(row, col, gradient_accumulator->W_2_cols)] = 69;
		}
	}
}

// This fucntion accumulates the temp gradient matricies from the Model struct into the gradient accumulator
void Accumulate_Gradient(Model* model, Gradient* gradient_accumulator) {
	// accumulate W_1_Grad
	for (int row = 0; row < model->W_1_rows; row++) {
		for (int col = 0; col < model->W_1_cols; col++) {
			gradient_accumulator->W_1_Grad[INDEX(row, col, model->W_1_cols)] += model->W_1_Grad[INDEX(row, col, model->W_1_cols)];
		}
	}

	// accumulate b_1_Grad
	for (int i = 0; i < model->b_1_cols; i++) {
		gradient_accumulator->b_1_Grad[i] += model->b_1_Grad[i];
	}

	// accumulate W_2_Grad
	for (int row = 0; row < model->W_2_rows; row++) {
		for (int col = 0; col < model->W_2_cols; col++) {
			gradient_accumulator->W_2_Grad[INDEX(row, col, model->W_1_cols)] += model->W_2_Grad[INDEX(row, col, model->W_2_cols)];
		}
	}

	// accumulate b_2_Grad
	for (int i = 0; i < model->b_2_cols; i++) {
		gradient_accumulator->b_2_Grad[i] += model->b_2_Grad[i];
	}
}

// this funciton applies the stochastic gradient descent learing rule to the model parameters in the Model struct. 
// The accumulated gradients are found within the gradient accumulator Gradient struct. 
void Stochastic_Gradient_Descent(Model* model, Gradient* gradient_accumulator, double learning_rate) {

	// apply learning rule to  W_1_Grad
	for (int row = 0; row < model->W_1_rows; row++) {
		for (int col = 0; col < model->W_1_cols; col++) {
			model->W_1[INDEX(row, col, model->W_1_cols)] -= learning_rate * gradient_accumulator->W_1_Grad[INDEX(row, col, model->W_1_cols)];
		}
	}

	// accumulate b_1_Grad
	for (int i = 0; i < model->b_1_cols; i++) {
		model->b_1[i] -= learning_rate * gradient_accumulator->b_1_Grad[i];
	}

	// apply learning rule to  W_1_Grad
	for (int row = 0; row < model->W_2_rows; row++) {
		for (int col = 0; col < model->W_2_cols; col++) {
			model->W_2[INDEX(row, col, model->W_2_cols)] -= learning_rate * gradient_accumulator->W_2_Grad[INDEX(row, col, model->W_2_cols)];
		}
	}

	// accumulate b_2_Grad
	for (int i = 0; i < model->b_2_cols; i++) {
		model->b_2[i] -= learning_rate * gradient_accumulator->b_2_Grad[i];
	}
}