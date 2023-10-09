#pragma once


/*  This macro is used to compute the index of a 2D matrix value that is stored
	as an abstraction wtihin a 1D array given the i,j index and num cols

	Here is an example of how you would index an integer from a matrix using INDEX:
	int element = matrix[INDEX(i, j, cols)];
*/
#define INDEX(i, j, cols) ((i) * (cols) + (j))



double* define_new_matrix(int num_rows, int nnum_cols);

void populate_matrix(int rows, int cols, int* matrix);

void matmul(int* C, int* A, int* B, int rows_A, int cols_A, int rows_B, int cols_B);

void display_matrix(int* matrix, int rows, int cols);