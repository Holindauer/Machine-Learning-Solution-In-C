#include "libraries.h"


//-------------------------------------------------------------------------------------------------------------------------<Weight Initialization>

// Function to generate a random float between two given values
double random_double(double min, double max) {
    return (max - min) * ((double)rand() / RAND_MAX) + min;
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
void he_initialize(double* matrix, int rows, int cols) {
    double stddev = sqrt(2.0 / cols);                                // get standard deviation for He initialization

    for (int i = 0; i < rows; i++) {                                // loop through matrix elements
        for (int j = 0; j < cols; j++) {

            // Box-Muller transform to approximate random numbers 

            double u1 = random_double(0.0, 1.0);                      // generate the numbers u1 and u2 
            double u2 = random_double(0.0, 1.0);                      // a from uniform distribution

            double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); // use u1 and u2 to compute z0 from 
                                                                    // the standard normal distribution
            // Assign the weight
            matrix[INDEX(i, j, cols)] = stddev * z0;                // multiply z0 by std to transform 
                                                                    // z0 into the correct distribution

        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------<Activation Functions>

/*
    This function appleis the ReLU activation function to an input matrix
    The Rectified Linear Unit is defined as f(x) = max(0, x). Which is the
    linear parent function above zero and a constant zero function less
    than and equal to 0.
*/
void ReLU(double* matrix, int rows, int cols)
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
void Softmax(double* matrix, int rows, int cols)
{
    // allocate memory to store logit values
    double* logits = (double*)malloc(rows * sizeof(double));

    // make a copy of the logits in logits array
    for (int i = 0; i < rows; i++)
    {
        logits[i] = matrix[INDEX(i, 1, cols)];
    }

    // initialize rolling summation for softmax computation
    double summation = 0;

    // apply softmax to matrix using logits array
    for (int i = 0; i < rows; i++)
    {
        // reset summation for exp of all logits
        summation = 0;

        for (int j = 0; j < rows; j++)
        {
            summation += exp(logits[i]);   // sum exp of all logits

        }

        // input softmax output into i'th logit
        matrix[INDEX(i, 1, cols)] = exp(logits[i]) / summation;
    }

    // free logits memory after use
    free(logits);
}


/*
    This function defines the forward pass for a 3 layer mlp.
    Provided weight matrices, num neurons per layer, input and
    its associated shape. The function outputs a matrix pointer
    array.
*/
double* forward_pass(double* W_1, double* W_2, double* W_3,            // weight matricies
    int layer_1_nodes, int layer_2_nodes, int layer_3_nodes,           // num neurons per layer
    double* X, int X_rows, int X_cols)                                 // input and input dimmensions
{

    int W_1_rows = X_rows,									            // set rows, cols							     
        W_1_cols = layer_1_nodes;

    int W_2_rows = layer_1_nodes,									    // set rows, cols
        W_2_cols = layer_2_nodes;

    int W_3_rows = layer_2_nodes,									    // set rows, cols
        W_3_cols = layer_3_nodes;


    double* layer_1_output = define_new_matrix(layer_1_nodes, 1);		// layer 1 output
    int layer_1_output_rows = layer_1_nodes,							// set rows, cols
        layer_1_output_cols = 1;

    double* layer_2_output = define_new_matrix(layer_2_nodes, 1);		// layer 2 output
    int layer_2_output_rows = layer_2_nodes,							// set rows, cols
        layer_2_output_cols = 1;

    double *layer_3_output = define_new_matrix(layer_3_nodes, 1);		// layer 3 output
    int layer_3_output_rows = layer_3_nodes,							// set rows, cols
        layer_3_output_cols = 1;



    // run forward pass: 

    matmul(layer_1_output, W_1, X, W_1_rows, W_1_cols, X_rows, X_cols);					// multiply input by weight matrix
    ReLU(layer_1_output, layer_1_output_rows, layer_1_output_cols);	                    // apply activation

    matmul(layer_2_output, W_2, layer_1_output, W_2_rows, W_2_cols, layer_1_nodes, 1);  // multiply layer 1 output by weight matrix
    ReLU(layer_2_output, layer_2_output_rows, layer_2_output_cols);					    // apply activation

    matmul(layer_3_output, W_3, layer_2_output, W_3_rows, W_3_cols, layer_2_nodes, 1);  // multiply layer 2 output by weight matrix
    Softmax(layer_3_output, layer_3_output_rows, layer_3_output_cols);                  // apply softmax activation for turning
                                                                                        // logits -----> probabilities

    // free memory after function call,
    //free(layer_1_output), free(layer_2_output), free(layer_3_output);

    return layer_3_output;

}



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

//-------------------------------------------------------------------------------------------------------------------------<Loss Functions>

double cross_entropy_loss(batch train_batch, int num_classes)
{
    int batch_size = train_batch.batch_size;           // extract batch size from batch

    double* labels = train_batch.labels;               // extract labels array from batch   

    double** predictions = train_batch.predictions;     // extract predictions array from batch

    double y_c = 0;                                    // initilize class/target relationship

    double example_loss = 0;                           // initialize loss for a single example

    double batch_loss = 0;                             // initialize loss for entire batch

    // compute cross entropy loss for the entire batch
    for (int N = 0; N < batch_size; N++)               // N is current example of batch
    {

        example_loss = 0;                              // reset loss for the current example

        for (int c = 0; c < num_classes; c++)          // c are the potential classes a digit example could be
        {
            if (labels[N] == c)  // check if current class of consideration is the N'th examples label
            {
                y_c = 1;         // if so, assign y_c = 1    
            }
            else                 // otherwise
            {
                y_c = 0;         // set it to 0
            }

            // compute loss for a single example
            example_loss += y_c * log(predictions[N][INDEX(c, 1, 1)]);
        }

        // add example loss to batch_loss rolling sum
        batch_loss += example_loss; 
    }

    // average batch_loss across num examples
    batch_loss /= (double)batch_size;


    return batch_loss;

}

