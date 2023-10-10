#include "libraries.h"



// Function to generate a random float between two given values
float random_float(float min, float max) {
    return (max - min) * ((float)rand() / RAND_MAX) + min;
}

/*
    Function to randomly initialize a weight matrix using He initialization.
    He initialization is useful for mitigating the vanishing/exploding grad
    problem. The goal is to keep the variance of activation function outputs
    roughly the same.

    For context, weight matricies have the shape [num_neurons, input_features]
    
    Weights are initialze from a normal distribution with a mean of 0 and a std
    of sqrt( 2 / input_features )
*/ 
void he_initialize(float* matrix, int rows, int cols) {
    float stddev = sqrt(2.0 / cols);                                // get standard deviation for He initialization

    for (int i = 0; i < rows; i++) {                                // loop through matrix elements
        for (int j = 0; j < cols; j++) {

            // Box-Muller transform to approximate random numbers 

            float u1 = random_float(0.0, 1.0);                      // generate the numbers u1 and u2 
            float u2 = random_float(0.0, 1.0);                      // a from uniform distribution

            float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); // use u1 and u2 to compute z0 from 
                                                                    // the standard normal distribution
            // Assign the weight
            matrix[INDEX(i, j, cols)] = stddev * z0;                // multiply z0 by std to transform 
                                                                    // z0 into the correct distribution

        }
    }
}


/*
    This function appleis the ReLU activation function to an input matrix
    The Rectified Linear Unit is defined as f(x) = max(0, x). Which is the 
    linear parent function above zero and a constant zero function less 
    than and equal to 0.
*/
void ReLU(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {                // iterate through matrix elements
        for (int j = 0; j < cols; j++) {
             
            if (matrix[INDEX(i, j, cols)] <= 0)     // check if 0 > x with regards to max(0,x)
            {
                matrix[INDEX(i, j, cols)] = 0;      // set to 0 if so
            }

        }

    }
}

/*
    This function defines the forward pass for a 3 layer mlp. 
    Provided weight matrices, num neurons per layer, input and 
    its associated shape. The function outputs a matrix pointer 
    array.
*/
double* forward_pass(double* W_1, double* W_2, double* W_3,        // weight matricies
    int layer_1_nodes, int layer_2_nodes, int layer_3_nodes,    // num neurons per layer
    double* X, int X_rows, int X_cols)                         // input and input dimmensions
{

    int W_1_rows = X_rows,									    // set rows, cols							     
        W_1_cols = layer_1_nodes;

    int W_2_rows = layer_1_nodes,									    // set rows, cols
        W_2_cols = layer_2_nodes;

    int W_3_rows = layer_2_nodes,									    // set rows, cols
        W_3_cols = layer_3_nodes;


    float* layer_1_output = define_new_matrix(layer_1_nodes, 1);		// layer 1 output
    int layer_1_output_rows = layer_1_nodes,							// set rows, cols
        layer_1_output_cols = 1;

    float* layer_2_output = define_new_matrix(layer_2_nodes, 1);		// layer 2 output
    int layer_2_output_rows = layer_2_nodes,							// set rows, cols
        layer_2_output_cols = 1;

    float* layer_3_output = define_new_matrix(layer_3_nodes, 1);		// layer 3 output
    int layer_3_output_rows = layer_3_nodes,							// set rows, cols
        layer_3_output_cols = 1;



    // run forward pass: 

    matmul(layer_1_output, W_1, X, W_1_rows, W_1_cols, X_rows, X_cols);					// multiply input by weight matrix
    ReLU(layer_1_output, layer_1_output_rows, layer_1_output_cols);	                    // apply activation

    matmul(layer_2_output, W_2, layer_1_output, W_2_rows, W_2_cols, layer_1_nodes, 1);  // multiply layer 1 output by weight matrix
    ReLU(layer_2_output, layer_2_output_rows, layer_2_output_cols);					    // apply activation

    matmul(layer_3_output, W_3, layer_2_output, W_3_rows, W_3_cols, layer_2_nodes, 1);  // multiply layer 2 output by weight matrix
    ReLU(layer_3_output, layer_3_output_rows, layer_3_output_cols);                     // apply activation




    // free memory after function call,
    //free(layer_1_output), free(layer_2_output), free(layer_3_output);

    return layer_3_output;

}