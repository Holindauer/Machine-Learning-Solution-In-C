#pragma once

#include "libraries.h" 


// Macro for computing strided 1D index of 2D array
#define INDEX(row, col, num_cols) ((row) * (num_cols) + (col))


// This struct is used to hold the model weights, states, and gradients through the 
// forward and backward pass.
typedef struct {
	// Matricies stored in memory contiguously -- use INDEX macro to access elements

	// Weights and Biases--- shape of weight matricies are determined by neurons x input
	double W_1[LAYER_1_NEURONS * NUM_FEATURES];
	double W_2[LAYER_2_NEURONS * LAYER_1_NEURONS];
	double b_1[LAYER_1_NEURONS];
	double b_2[LAYER_2_NEURONS];


	// Gradients -- in order for backpropagation. All gradients listed below  
	// are of the loss with respect to the vector/matrix in the variable name.

	// NOTE: these gradients are temp gradients that will only hold the gradient of 
	// the loss for a single input. The accumulation of gradients over the entire 
	// training set is done within the Gradient struct.

	double output_Grad[LAYER_2_NEURONS];
	double logits_Grad[LAYER_2_NEURONS]; 
	double W_2_Grad[LAYER_2_NEURONS * LAYER_1_NEURONS];
	double b_2_Grad[LAYER_2_NEURONS];

	double hidden_Grad[LAYER_1_NEURONS];
	double W_1_Grad[LAYER_1_NEURONS * NUM_FEATURES];
	double b_1_Grad[LAYER_1_NEURONS];
	

	// Network States -- needed for backprop
	double input[NUM_EXAMPLES][NUM_FEATURES];
	double hidden[NUM_EXAMPLES][LAYER_1_NEURONS];  
	double logits[NUM_EXAMPLES][LAYER_2_NEURONS];  // <-- pre softmax
	double output[NUM_EXAMPLES][LAYER_2_NEURONS];  // <-- post softmax
	double prediction[NUM_EXAMPLES];               // <-- post argmax

	// Matrix Shapes
	int W_1_rows, W_1_cols;
	int W_2_rows, W_2_cols; 
	int b_1_cols;
	int b_2_cols;

}Model;


typedef struct {

	// Matricies stored in memory contiguously -- use INDEX macro to access elements
	double W_2_Grad[LAYER_2_NEURONS * LAYER_1_NEURONS];
	double b_2_Grad[LAYER_2_NEURONS];
	double W_1_Grad[LAYER_1_NEURONS * NUM_FEATURES];
	double b_1_Grad[LAYER_1_NEURONS];

	// Matrix Shapes
	int W_1_rows, W_1_cols;
	int W_2_rows, W_2_cols;
	int b_1_cols;
	int b_2_cols;

}Gradient;



typedef struct {

	// features
	double features[150][4]; // iris: 150 examples, 4 features each
	double targets[150][3];     // 3 classes, 1 hot encoded

}Data;