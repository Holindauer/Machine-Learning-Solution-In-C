#pragma once

#include "libraries.h"

/*
	This struct will be used to house a single example of the mnist datset. 
	double* image will hold a flattened mnist image. int label holds that 
	image's label. 

	To hold the entire dataset for training, an array of example structs will
	be created. 
*/

typedef struct {
	double* image;
	int label;
}example;


/*
	This struct is used to store the weight matricies as of a neural network.

	Along with matricies for one hidden state and the gradient of the cost 
	w.r.t. each weight matrix of the network.

	Indexing of the matricies should be done with the INDEX macro.
*/
typedef struct {
	//----------------------------------------------------Weight Matricies

	double* W_1;             // layer 1 weights
	double* W_1_grad;        // grad w.r.t W_1
	int W_1_rows, W_1_cols;  

	double* b_1;             // layer 1 bias 
	double* b_1_grad;        // grad w.r.t. b_1
	int b_1_rows, b_1_cols;  

	double** pre_activations_1;
	double** hidden;         // array of hidden state arrays
	int batch_size;          // batch_sizze num arrays in hidden 


	double* W_2;             // layer 2 weights
	double* W_2_grad;        // grad w.r.t. W_2
	int W_2_rows, W_2_cols;     

	double* b_2;             // layer 2 bias 
	double* b_2_grad;        // gradient w.r.t. b_2
	int b_2_rows, b_2_cols;
}network;


/*
	This struct is used to create an array of arrays that will
	hold the model outputs of a single batch.
*/
typedef struct {

	double* input_vector;

	double* output_vector;  // holds an output vector for a single example
	                                        // layer 2 neurons is the output size

	double target;                          // true target for a single predicted example


}outputs;

