#include "functions.h"
#include "structs.h"
#include "libraries.h"

// this function computes the argmax of an array
int argmax_vector(double* vector, int vector_length) {

	int	max_idx = 0;
	double max = 0;

	for (int i = 0; i < vector_length; i++) {

		if (vector[i] > max) {
			max_idx = i;
			max = vector[i];
		}
	}
	return max_idx;
}


double rand_double() {
	return (double)rand() / (RAND_MAX + 1.0);
}