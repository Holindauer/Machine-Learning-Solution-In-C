#include "functions.h"
#include "structs.h"
#include "libraries.h"


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
    // Iterate through each row to apply softmax independently
    for (int i = 0; i < rows; i++)
    {
        // Find the maximum logit to improve numerical stability
        double max_logit = matrix[INDEX(i, 0, cols)];
        for (int j = 1; j < cols; j++)
        {
            if (matrix[INDEX(i, j, cols)] > max_logit)
            {
                max_logit = matrix[INDEX(i, j, cols)];
            }
        }

        // Compute the sum of exponentials of the logits
        double sum_exp = 0.0;
        for (int j = 0; j < cols; j++)
        {
            sum_exp += exp(matrix[INDEX(i, j, cols)] - max_logit);
        }

        // Apply the softmax function to scale the logits
        for (int j = 0; j < cols; j++)
        {
            matrix[INDEX(i, j, cols)] = exp(matrix[INDEX(i, j, cols)] - max_logit) / sum_exp;
        }
    }
}