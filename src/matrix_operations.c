#include "functions.h"
#include "structs.h"
#include "libraries.h"


/*
	create_matrix() allocates memory for a matrix
	of a given number of rows and columns. values of
	the matrix are initialize to 0;
*/
double* create_matrix(int rows, int cols)
{
	// allocate memory for matrix

	double* matrix = (double*)malloc(rows * cols * sizeof(double));


	if (matrix == NULL)  // error handling
	{
		exit(1);
	}

	// Initialize memory to zero
	memset(matrix, 1, rows * cols * sizeof(double));

	return matrix;
}


/*
	This function displays a matrix row by row.
*/
void display_matrix(double* matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		printf("\n");

		for (int j = 0; j < cols; j++)
		{
			// print ith,jth cols of the matrix
			printf(" %.2f ", matrix[INDEX(i, j, cols)]);
		}
	}
}


/*
	This function computes the matrix multiplication of a matrix A and B.

	As a preconditions, a matrix C of the correct shape of AB should already
	be created and it's address passed into the function. Along with A, B, and
	their associeated shapes

	The value of the matrix are initialize all to 0.
*/
void matmul(double* C, double* A, double* B, int rows_A, int cols_A, int rows_B, int cols_B)
{

	// perform matmul alg: C_ij = n_Sigma_k=1 A_ik * B_kj

	// intialize var to stor element at index i, j in matrix C  
	double element_ij = 0;

	// iterate through the rows of Matrix A
	for (int i = 0; i < rows_A; i++)
	{
		// iterate through the columns in B
		for (int j = 0; j < cols_B; j++)
		{
			// reset dot product
			element_ij = 0;

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

/*
	This function performs an elementwise matrix addition with two matricies, A and B. 
	Where B stands for Bias. This function is used within the forward() fucntion within
	model.c. 

	The like elements of matrix B are added directly into matrix A.

*/
void add_bias(double* A, double* B, int rows, int cols)
{
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			A[INDEX(row, col, cols)] += B[INDEX(row, col, cols)];
		}
	}

}