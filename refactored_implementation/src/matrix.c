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


void Display_Matrix(double* matrix, int rows, int cols) {
	printf("\n");
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			printf("%.2f ", matrix[INDEX(row, col, cols)]);
		}
		printf("\n"); // This will print a new line after each row for better readability
	}
}
