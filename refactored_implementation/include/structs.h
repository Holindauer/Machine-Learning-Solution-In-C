#pragma once

#include "libraries.h" 


// Macro for computing strided 1D index of 2D array
#define INDEX(row, col, num_cols) ((row) * (num_cols) + (col))

typedef struct {
	// Matricies stored in memory contiguously -- use INDEX macro to access elements

	// Weights --- shape of weight matricies are determined by neurons x input
	double W_1[LAYER_1_NEURONS * NUM_FEATURES];
	double W_2[LAYER_2_NEURONS * LAYER_1_NEURONS];

	// Biases
	double b_1[LAYER_1_NEURONS];
	double b_2[LAYER_2_NEURONS];


	// Garients
	double W_1_Grad[LAYER_1_NEURONS * NUM_FEATURES];
	double W_2_Grad[LAYER_2_NEURONS * LAYER_1_NEURONS];
	double b_1_Grad[LAYER_1_NEURONS];
	double b_2_Grad[LAYER_2_NEURONS];

	// Network States
	double input[NUM_EXAMPLES][NUM_FEATURES];  // <-- 2D array used for 'batch' of vectors 
	double hidden[NUM_EXAMPLES][LAYER_1_NEURONS];
	double output[NUM_EXAMPLES][LAYER_2_NEURONS];

	int prediction[NUM_EXAMPLES];

	// Matrix Shapes
	int W_1_rows, W_1_cols;
	int W_2_rows, W_2_cols; 
	int b_1_cols;
	int b_2_cols;

}Model;



typedef struct {

	// features
	double features[150][4]; // iris: 150 examples, 4 features each
	double targets[150][3];     // 3 classes, 1 hot encoded

}Data;