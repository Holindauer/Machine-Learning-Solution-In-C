#pragma once

#include "structs.h"

//--------------------------------------------------------------------------------------weight_initialization.c

/* computes a random double within range parametrized by min and max */
double random_double(double min, double max);

/* This function is used to initialized the model paramteters using he initialization */
void he_initialize(double* weight_matrix, int rows, int cols);

//--------------------------------------------------------------------------------------matrix_operations.c

/* allocates memory for a matrix based on rows, cols params */
double* create_matrix(int rows, int cols);

/* This function displays a matrix to terminal */
void display_matrix(double* matrix, int rows, int cols);

/*  This function computes the matmul C = AB given the each
	preinitialized matrix and their associeate shapes */
void matmul(double* C, double* A, double* B, int rows_A, int cols_A, int rows_B, int cols_B);

/* Adds bias to a matrix w/ elementwise addition -- used within forward() pass */
void add_bias(double* A, double* B, int rows, int cols);

/*
This function computes the transpose of a matrix i.e. swaps rows, cols

		A = [ 1, 2, 3]
			[ 4, 5, 6]

		A_T = [1, 4]
			  [2, 5]
			  [3. 6]

	Array A_T must have the same elements as Array A and should be indexed
	using the roversed rows, cols of A
*/
void transpose(double* A_T, double* A, int A_rows, int A_cols);

//--------------------------------------------------------------------------------------activation_functions.c

/* This function computes ReLU() elementwise on a flattened matrix array */
void ReLU(double* matrix, int rows, int cols);

/* This function computes Softmax() elementwise on a flattened matrix array */
void Softmax(double* vector, int length);


//--------------------------------------------------------------------------------------model.c

/* This function allocates mem for and initializes all elements within the weights struct */
void init_model(weights* net);

/* This function computes the forward pass of the model */
void forward(weights net, double* example, double* model_output, int batch_element);

/* Thif function perfroms argmax() on the probability evctor output of the model -- returns double */
double predict(double* model_output);

//--------------------------------------------------------------------------------------memory.c

/* This fucntion handles memory allocation errors */
void check_memory_allocation(double* arr);

/* This fucntion frees the memory within the datset struct array */
void free_dataset(example* dataset, int num_examples);

/* this fucntion initializes a declared array to zero */
void arr_init_zero(double* arr, int num_elements);

/* This fucntion intitalizes each nested array within a batch_outputs array */
void init_batches(batch_outputs* batch, int batch_size);

/* This function initializes each hidden state array within the network */
void init_hidden(weights* net);

/* This function frees the weights of the weights struct */
void free_network(weights* net);

//--------------------------------------------------------------------------------------load_data.c

/* This function loads in flattened mnist digits from a csv */
void load_data(const char* filename, example* dataset, int num_digits);

/* inititalizes each array in example in array of example stracts */
void initialize_dataset(example* dataset, int num_examples);

//--------------------------------------------------------------------------------------loss_functions.c

/*
	This function computes the cross entropy loss for a batch during training.
	The batch struct contains both the example matrix, label array, and predictions
	array within its members.

	cross entropy for a single datapoint is defined below as:

	L(y, y_hat) = - M_sigma_c=1 (y_c * log(y_hat_c) )

	Where:  M is the number of classes

			M_sigma_c=1 is the summation starting at 1 to the num classes

			y_c is a binary value indicating whether the target label and the
			c'th label for the current example are the same.

			y_hat_c is the the model's predicted probability of whether the
			current example belongs to class c

	Thus, cross entropy for the entire batch is:

	batch_loss = -(1/N) N_sigma_i=1 ( L(y, y_hat) )

	Where N is the number of examples in the batch
*/
double cross_entropy_loss(batch_outputs* outputs, int batch_size);

//--------------------------------------------------------------------------------------backprop.c

/*
	 This function computes the derivative of ReLU for a given input to the ReLU() func

	 ReLU() = f(x) = max(0, x)

	 f'(x) = 0    when x <= 0   ---  f'(x) = 1    when x > 0
*/
void ReLU_derivative(double* pre_activations, double* computed_derivatives, int num_elements);

/*
	This function accumulates the gradient of the cost with respect to each weight for the
	weight matrix of the last layer of the model.

	dL/dWij = (dL/dz_i) * (dz_i/dW_ij)         <--- To understand the derivation of this
			= (p_i - y_i) * h_j                     look at the note above the function W_2
													and b_2 grad comp functions in backprop.c


	The function takes the follwing arguments:

		- weights* net is a pointer to the weights struct of model

		- batch_outputs* batch is an array of batch_output structs whose members contains the
							   following data for each example predicted on within a batch:

			 - the output vector of of the model, containing the likelihood each class is the target
			 - as well as the label for each example within the array
*/
void backprop_W_2(weights* net, batch_outputs* batch);

/*
	This function accumulates the gradient of the cost with respect to each weight for the
	weight bias vector of the last layer of the model.

	dL/dbi = p_i - y_i      <--- To understand the derivation of this
								 look at the note above the function W_2
								 and b_2 grad comp functions in backprop.c


	The function takes the follwing arguments:

		- weights* net is a pointer to the weights struct of model

		- batch_outputs* batch is an array of batch_output structs whose members contains the
							   following data for each example predicted on within a batch:

			 - the output vector of of the model, containing the likelihood each class is the target
			 - as well as the label for each example within the array
*/
void backprop_b_2(weights* net, batch_outputs* batch);