#include "libraries.h"



/*
	given a number of rows and columns, this function will allocate memory to
	a flattened matrix, check for a malloc() error, and return the matrix.
*/
double* define_new_matrix(int num_rows, int num_cols)
{
	/*
		Use malloc to initilize memory addresses for a flattened matrix.

		rows x cols x sizeof(int) computes the number of bytes needed to
		store the amount of integers within a rows x cols matrix

		matrix contains pointers to ineger types
	*/
	double* matrix = (int*)malloc(num_rows * num_cols * sizeof(int));

	if (matrix == NULL)  // if malloc fails, the first pointer will be NULL
	{
		exit(1);        // stop the program in the case of a malloc() failure
	}

	return matrix;

}



/*
	This function computes the matrix multiplication of a matrix a and B.
	A matrix C is created, filled with the appropriate values, and returned.
	Preconditions: Matrix A and B exist.
*/
void matmul(int* C, int* A, int* B, int rows_A, int cols_A, int rows_B, int cols_B)
{

	// perform matmul alg: C_ij = n_Sigma_k=1 A_ik * B_kj

	// intialize var to stor element at index i, j in matrix C  
	int element_ij = 0;

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
	This function displays a matrix.
	Precondition, matrix exists.
*/
void display_matrix(int* matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		printf("\n");

		for (int j = 0; j < cols; j++)
		{
			// print ith,jth cols of the matrix
			printf(" %d ", matrix[INDEX(i, j, cols)]);
		}
	}


}