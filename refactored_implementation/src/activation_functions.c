#include "functions.h"
#include "structs.h"
#include "libraries.h"

// This funciton performs Elementwise ReLU activation on a matrix -- ReLU = max{0, x}
Elementwise_ReLU(double* matrix, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		if (matrix[i] < 0) { //ReLU = max{0, x}
			matrix[i] = 0;
		}
	}
}

/*
	The softmax function normalizes the input vector into a probability distribution, where
	each element of the output vector represents the probability of the corresponding class,
	and the sum of all these probabilities is 1.

	for a vector x = [x_1, x_2, ...., x_n]
	softmax(x)_i = exp(x_i) / n_sigma_j=1 [exp(x_j)]
*/
void Elementwise_Softmax(double* matrix, int vector_length) {
    // Find the maximum element to prevent overflow in exp
    double maxElement = matrix[0];
    for (int i = 1; i < vector_length; i++) {
        if (matrix[i] > maxElement) {
            maxElement = matrix[i];
        }
    }

    // Compute denominator of softmax(x)_i and apply maxElement subtraction for numerical stability
    double sum = 0;
    for (int i = 0; i < vector_length; i++) {
        matrix[i] = exp(matrix[i] - maxElement); // Subtract maxElement for stability
        sum += matrix[i];
    }

    // Epsilon to prevent division by zero
    double epsilon = 1e-8;
    sum = sum < epsilon ? epsilon : sum;

    // Compute softmax over terms
    for (int i = 0; i < vector_length; i++) {
        matrix[i] /= sum;
    }
}
