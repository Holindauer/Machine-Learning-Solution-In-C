#include "structs.h"
#include "functions.h"
#include "libraries.h"

/*
	Source matricies A and B mulitplied to produce the destination matrix
	Integer Parameters M, N, P correspones to the shape of AB: [M x N] * [N x P]
*/
void MatMul(double* dest_matrix, double* src_A, double* src_B, int A_rows, int A_cols, int B_cols) {
	
	// c_ij = n_sigma_k=1 [aik bkj]
	for (int i = 0; i < A_rows; i++) {
		for (int j = 0; j < B_cols; j++) {
			for (int k = 0; k < A_cols; k++) {
				dest_matrix[INDEX(i, j, B_cols)] += src_A[INDEX(i, k, A_cols)] * src_B[INDEX(k, j, B_cols)]; // <--- dot of i'th col and j'th row
			}
		}
	}
}

/*
	This func performs in place elementwise addition matricies
	and vectors must tbe the same shape for elementwise addition
*/
void Elementwise_Addition(double* destination, double* matrix_to_add, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		destination[i] += matrix_to_add[i];
	}
}

/*
	This func performs in place elementwise addition matricies
	and vectors must tbe the same shape for elementwise addition
*/
void Elementwise_Subtraction(double* destination, double* matrix_to_subtract, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		destination[i] -= matrix_to_subtract[i];
	}
}

// This function performs in-place element-wise multiplication
void Elementwise_Multiply(double* destination, double* matrix1, double* matrix2, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		destination[i] = matrix1[i] * matrix2[i];
	}
}

void Copy_Matrix(double* destination, double* matrix_to_copy, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		destination[i] = matrix_to_copy[i];
	}
}




void Display_Matrix(double* matrix, int rows, int cols) {
	printf("\n");
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			printf("%.2f ", matrix[INDEX(row, col, cols)]);
		}
		printf("\n"); // This will print a new line after each row for better readability
	}
}


/*
This function computes the transpose of a matrix i.e. swaps rows, cols

		A =  1 2 3
			 4 5 6

		A_T = 1 4
			  2 5
			  3 6

	Array A_T must have the same elements as Array A and should be indexed
	using the roversed rows, cols of A
*/
void Transpose_Matrix(double* destination_matrix, double* src_matrix, int src_rows, int src_cols)
{
	for (int row = 0; row < src_rows; row++) {    
		for (int col = 0; col < src_cols; col++)
		{
			destination_matrix[INDEX(col, row, src_rows)] = src_matrix[INDEX(row, col, src_cols)];  // swaps rows with cols
		}
	}
}


void Zero_Matrix(double* matrix, int rows, int cols) {
	for (int i = 0; i < (rows * cols); i++) {
		matrix[i] = 0;
	}
}