#include "libraries.h"
#include "functions.h"
#include "structs.h"


void test_matmul(void) {

	double matrix[3][3] = {
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8}
	};

	Display_Matrix(matrix, 3, 3);

	double mul[3][3] = { 0 };

	MatMul(mul, matrix, matrix, 3, 3, 3);
	Display_Matrix(mul, 3, 3);

}