#include "matmul.h"

/*
	This program contains computes the matrix multiplication for a 
	matrix of dummy integer values of two matricies of variable size 
*/

int main(void) 

{

	// ------------------------- create first matrix -------------------------

	int A_rows = 5, A_cols = 7;						   // intialize rows and cols

	int* A = define_new_matrix(A_rows, A_cols);      // create matrix  A
	
	populate_matrix(A_rows, A_cols, A);			   // populate matrix with dummy values
	
	printf("\n\nA = \n");
	display_matrix(A, A_rows, A_cols);	        // display

	// ------------------------- create second matrix -------------------------

	int B_rows = 7, B_cols = 10;				      // intialize rows and cols

	int* B = define_new_matrix(B_rows, B_cols);     // create matrix  B

	populate_matrix(B_rows, B_cols, B);			  // populate matrix with dummy values

	printf("\n\nB = \n");
	display_matrix(B, B_rows, B_cols);		   // display

	// ------------------------------ MatMul ----------------------------------

	/*  define a new matrix with define_new)_matrix() to fill with
		the A multiplied by B The shape is rows_A by cols_B. It must
		be define in main to avoid a dangling pointer error if returned 
		by matmul
	*/
	int* C = define_new_matrix(A_rows, B_cols);
	

	matmul(C, A, B, A_rows, A_cols, B_rows, B_cols);  // perform matrix multiplication


	printf("\n\nC = AB = \n");
	display_matrix(C, A_rows, B_cols);             // display

	return 0;
}