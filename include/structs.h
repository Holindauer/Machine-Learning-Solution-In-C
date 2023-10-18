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
	This struct is used to store the weight matricies of a neural network.
	Indexing of the matricies should be done with the INDEX macro.
*/
typedef struct {
	//----------------------------------------------------Weight Matricies

	double* W_1;             // layer 1 weight matrix
	int W_1_rows, W_1_cols;  

	double* b_1;              // layer 1 bias matrix
	int b_1_rows, b_1_cols;  


	double* W_2;             // layer 2 weight matrix
	int W_2_rows, W_2_cols;     

	double* b_2;             // layer 2 bias matrix
	int b_2_rows, b_2_cols;




}weights;


/*
	This struct is used to create an array of arrays that will
	hold the model outputs of a single batch.
*/
typedef struct {

	double output_vector[LAYER_2_NEURONS];  // holds an output vector for a single example
	                                        // layer 2 neurons is the output size

	double target;                          // true target for a single predicted example


}batch_outputs;