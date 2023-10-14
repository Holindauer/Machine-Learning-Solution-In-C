#pragma once

#include "libraries.h"
#include "structs.h"

//------------------------------------------------------------------------ load_data.c

/*
    load_data() loads in mnist digits from a csv file where each row of the csv is
    a flattened image. The first column is the label of the image and the subsequent
    columns correspond to the flattened pixel intensities.

    load_data() takes two arguments. The first is const char* filename which is a
    string containing the name of the csv file holding the data. The second arg is
    int num_digits, which is the number of examples that will be extracted from the
    dataset.
*/
dataset load_data(const char* filename, int num_digits);

/*
    This function frees the memory allocated
    for the dataset after done with it
*/
void free_data(dataset* data, int num_examples);

//------------------------------------------------------------------------ matrix_operations.c

/*
    create_matrix() allocates memory for a matrix
    of a given number of rows and columns
*/
double* create_matrix(int rows, int cols);

/*
    This function displays a matrix row by row.
*/
void display_matrix(double* matrix, int rows, int cols);

/*
    This function computes the matrix multiplication of a matrix A and B.

    As a preconditions, a matrix C of the correct shape of AB should already
    be created and it's address passed into the function. Along with A, B, and
    their associeated shapes
*/
void matmul(double* C, double* A, double* B, int rows_A, int cols_A, int rows_B, int cols_B);

//------------------------------------------------------------------------ model.c

/*
    This function performs a forward pass using
    the weights within a network_weights struct
*/
double* forward_pass(network_weights net, double* input_example);

/*
    This function frees the weight matricies of an mlp when done
*/
void free_model_weights(network_weights* network);

//------------------------------------------------------------------------ initialize_weights.c

/*
    Function to generate a random float between two given values
*/ 
double random_double(double min, double max);

/*
    Function to randomly initialize a weight matrix using He initialization.
    He initialization is useful for mitigating the vanishing/exploding grad
    problem. The goal is to keep the variance of activation function outputs
    roughly the same.

    For context, weight matricies have the shape [num_neurons, input_features]

    Weights are initialze from a normal distribution with a mean of 0 and a std
    of sqrt( 2 / input_features )
*/
void he_initialize(double* matrix, int rows, int cols);


//------------------------------------------------------------------------ activation_functions.c

/*
    This function appleis the ReLU activation function to an input matrix
    The Rectified Linear Unit is defined as f(x) = max(0, x). Which is the
    linear parent function above zero and a constant zero function less
    than and equal to 0.
*/
void ReLU(double* matrix, int rows, int cols);


/*
    This function takes a matrix along with its dimmensions as input parameters
    and applies the softmax activation funtion to it.

    The softmax function is used to turn the raw outputs of a network for multi
    class classification into probabilitiy values that add up to 1.

    Thus, the matrix input to this function is a [num_classes, 1] matrix. Where,
    in the context of mnist, num_class=10 and the each element of the matrix
    corresponds to the logit value for each digit/index.


    Softmax for the i'th logit value of a logit vector is: exp(y_i) / sigma_j ( exp(y_j) )

    Where: y_i is the i'th classes logit and sigma_j ( exp(y_j) ) is the sum of all logits.
*/
void Softmax(double* matrix, int rows, int cols);

//------------------------------------------------------------------------ loss_functions.c