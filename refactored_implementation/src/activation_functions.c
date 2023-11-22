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