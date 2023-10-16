#include "functions.h"
#include "structs.h"
#include "libraries.h"


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



void Softmax(double* vector, int length)
{
    // Find the maximum logit to improve numerical stability
    double max_logit = vector[0];
    for (int i = 1; i < length; i++)
    {
        if (vector[i] > max_logit)
        {
            max_logit = vector[i];
        }
    }

    // Compute the sum of exponentials of the logits
    double sum_exp = 0.0;
    for (int i = 0; i < length; i++)
    {
        sum_exp += exp(vector[i] - max_logit);
    }

    // Apply the softmax function to scale the logits
    for (int i = 0; i < length; i++)
    {
        vector[i] = exp(vector[i] - max_logit) / sum_exp;
    }
}
