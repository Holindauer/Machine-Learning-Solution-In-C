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

/*
	This function computes the outer product of two vectors,

	it takes as inpute two vectors to computes the outer product of,
	their lengths, and an outer_product array to store the resultant
	matrix

	The outer product is defined as:

	C = a [outer product ] b
	C_ij = a_i * b_j

	Where: C is an mxn matrix
		   a is an m dimmensional vector
		   b is an n dimmesnional vector
*/
void outer_product(double* a, double* b, double* C, int a_len, int b_len);


//--------------------------------------------------------------------------------------activation_functions.c

/* This function computes ReLU() elementwise on a flattened matrix array */
void ReLU(double* matrix, int rows, int cols);

/* This function computes Softmax() elementwise on a flattened matrix array */
void Softmax(double* vector, int length);


//--------------------------------------------------------------------------------------model.c

/* This function allocates mem for and initializes all elements within the weights struct */
void init_model(network* net);

/* This function computes the forward pass of the model */
void forward(network net, double* example, double* model_output, int batch_element);

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
void init_batches(outputs* batch, int batch_size);

/* This function initializes each hidden state array within the network */
void init_hidden(network* net);

/* This function frees the weights of the weights struct */
void free_network(network* net);

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
double cross_entropy_loss(outputs* outputs, int batch_size);

//--------------------------------------------------------------------------------------backprop.c

/*
	This the master function that function fascilliates backpropagation for
	the gradient of the loss with respect to the parameters of the network.
	It is specific to the following architecture:

		hidden = ReLU(W_1 * input + b_1)
		output = Softmax(W_2 * hidden + b_2)

	With the follwoing shapes:

		input; [784, 1]

		W_1: [128, 784]
		b_1: [128, 1]

		hidden: [128, 1]

		W_2: [10, 128]
		b_2: [10, 1]


	The function works by computing the gradient of the cross entropy with
	respect to each parameter of the model for each example of a batch. Thse
	gradients are accumulated and then averaged across the batch.

*/
void backprop(network* net, outputs* batch);

/*
	This function computes the gradient of the loss w.r.t. the weight matrix
	of the output later, this is the first function to be called in the backprop()
	master function.

	backprop_W_2() retruns the gradient of lsos w.r.t. the pre activations of the
	hidden layer, z_2. dL/dz_2 is need for the the subsequent function calls in the
	backprop() master function. dL_dz_2 is also the equivalent to the grad of J()
	w.r.t. b_2

	The following mathematical computations are performed in this function:

		dL/dz_i = p_i - y_i

		dL/dWij = (dL/dz_i) * (dz_i/dW_ij)
				= (p_i - y_i) [outer propduct] h_j

		Written equivalently as:

		dL/dW_2 = dL/dz_2 [outer product] hidden
		dL/db_2 = dL/dz_2

	to understand the derivation of these formulas, check the ReadME
*/
double* accumulate_W_2_grad(network* net, outputs* batch, int batch_element);

/*
	This function accumulates the gradient of the cost J() w.r.t.
	the bias vector the the output layer. This grad has already
	been computed within backprop_W_2(), so all that must happed
	is to accumulated into net.b_2_grad.

	dL/db_2 = dL/dz_2 <--- whihc must be passed in as an arg
*/
void accumulate_b_2_grad(network* net, double* dL_dz_2);

/*
	This function propagates the gradient of the cost w.r.t. the weight matrix
	of the hiddne layer for a single example wihtin a batch. It takes as input
	dL/dz_2, which was computed in accumulate_W_2_grad.

	The func outputs dL_dz_1 which is used within accumulate_b_1_grad

*/
double* accumulate_W_1_grad(network* net, outputs* batch, double* dL_dz_2, int batch_element);

/*
	This function accumulates the gradient w.r.t. the bias parameters of the
	input layer. The grad has already been precomputed in accumulate_W_1_grad,

		dL_db_i = dL_dz_1
*/
void accumulate_b_1_grad(network* net, outputs* batch, double* dL_dz_1, int batch_element);

//--------------------------------------------------------------------------------------SGD.c

/*
	This function performs the gradient descent learning rule after
	the gradients of a batch have been computed.

	The gradients and weights are all held inside of the network struct,
	which is passed in as an argument. The other parameter needed is the
	learning rate, which is to be set before the training loop begins
	within main().

	The gradient descent learning rule is defined as:

	the update for a parameter theta_i = theta_i - ( learning_rate * dL/theta_i)
*/
void gradient_descent(network* net, double learning_rate);