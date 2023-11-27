#include "functions.h"
#include "structs.h"
#include "libraries.h"


/* allocates memory for a matrix based on rows, cols params */
double* create_matrix(int rows, int cols)
{
	double* matrix = (double*)malloc(rows * cols * sizeof(double));
	if (matrix == NULL){ exit(1);}

	memset(matrix, 1, rows * cols * sizeof(double)); // iniitalize mem to 1

	return matrix;
}

/* This function displays a matrix to terminal */
void display_matrix(double* matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		printf("\n");

		for (int j = 0; j < cols; j++)
		{
			printf(" %.2f ", matrix[INDEX(i, j, cols)]);
		}
	}
}

/*  This function computes the matmul C = AB given the each 
	preinitialized matrix and their associeate shapes */
void matmul(double* C, double* A, double* B, int rows_A, int cols_A, int rows_B, int cols_B)
{
	double element_ij = 0;

	for (int i = 0; i < rows_A; i++)  {     // iterate through rows
		for (int j = 0; j < cols_B; j++){   // iterate through the columns in B
	
			element_ij = 0; // reset dot product

			// compute dot product of the ith row and jth col
			for (int k = 0; k < cols_A; k++)
			{
				//  rolling sum of element wise multiplications 
				element_ij += A[INDEX(i, k, cols_A)] * B[INDEX(k, j, cols_B)];
			}

			// place dot product of the ith row and jth col into C(i, j)
			C[INDEX(i, j, cols_B)] = element_ij;
		}
	}
}

/* Adds bias to a matrix w/ elementwise addition -- used within forward() pass */
void add_bias(double* A, double* B, int rows, int cols)
{
	for (int row = 0; row < rows; row++){
		for (int col = 0; col < cols; col++)
		{
			A[INDEX(row, col, cols)] += B[INDEX(row, col, cols)];  // add bias into ij'th element
		}
	}
}


/* 
This function computes the transpose of a matrix i.e. swaps rows, cols

		A = [ 1, 2, 3]
			[ 4, 5, 6]

		A_T = [1, 4]
			  [2, 5]
			  [3. 6]

	Array A_T must have the same elements as Array A and should be indexed
	using the roversed rows, cols of A
*/
void transpose(double* A_T, double* A, int A_rows, int A_cols)
{
	for (int row = 0; row < A_rows; row++){     // iterate through A
		for (int col = 0; col < A_cols; col++)
		{
			A_T[INDEX(col, row, A_rows)] = A[INDEX(row, col, A_cols)];  // swaps rows with cols
		}
	}
}


/*
	This function computes the outer product of two vectors,
	
	it takes as inpute two vectors to computes the outer product of, 
	their lengths, and an outer_product array to store the resultant 
	matrix 

	The outer product is defined as: 

	C = a [outer product ] b   
	C_ij = a_i * b_j

	Where: C is an mxn matrix
	       a is an m dimmensional vector
		   b is an n dimmesnional vector 
*/
void outer_product(double* a, double* b, double* C, int a_len, int b_len)
{
	for (int i = 0; i < a_len; i++) {
		for (int j = 0; j < b_len; j++) {
			C[INDEX(i, j, b_len)] = a[i] * b[j];  // C_ij = a_i * b_j
		}
	}
}