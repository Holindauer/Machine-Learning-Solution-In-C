#include "functions.h"
#include "structs.h"
#include "libraries.h"


/* 
	This the master function that function fascilliates backpropagation for 
	the gradient of the loss with respect to the parameters of the network. 
	It is specific to the following architecture:

		hidden = ReLU(W_1 * input + b_1)
		output = Softmax(W_2 * hidden + b_2)

	With the follwoing shapes:

		input; [784, 1]
		
		W_1: [128, 784] 
		b_1: [128, 1]

		hidden: [128, 1]

		W_2: [10, 128]
		b_2: [10, 1]


	The function works by computing the gradient of the cross entropy with 
	respect to each parameter of the model for each example of a batch. Thse
	gradients are accumulated and then averaged across the batch.

*/
void backprop(network* net, outputs* batch) 
{	
	// zero gradients
	arr_init_zero(net->W_2_grad, W_2_ELEMENTS);   
	arr_init_zero(net->W_1_grad, W_1_ELEMENTS);
	arr_init_zero(net->b_2_grad, LAYER_2_NEURONS);
	arr_init_zero(net->b_1_grad, LAYER_1_NEURONS);

	// initialize temp arrays 
	double* dL_dz_2, *dL_dz_1;  

	for (int b = 0; b < net->batch_size; b++)
	{
		// Propagate grad through output layer 

		dL_dz_2 = accumulate_W_2_grad(net, batch, b);  // <-- gradients are accumualted into the network struct internally
		accumulate_b_2_grad(net, dL_dz_2); 


		// propagate grad through the hidden layer
		dL_dz_1 = accumulate_W_1_grad(net, batch, dL_dz_2, b);
		accumulate_b_1_grad(net, batch, dL_dz_1, b);

		// free temp grad arrays when done
		free(dL_dz_2); 
		free(dL_dz_1);
	}

	// average grads over batch_size

	for (int i = 0; i < W_2_ELEMENTS; i++) { net->W_2_grad[i] /= (double)net->batch_size; }    // W_2
	for (int i = 0; i < LAYER_2_NEURONS; i++) { net->b_2_grad[i] /= (double)net->batch_size; } // b_2

	for (int i = 0; i < W_1_ELEMENTS; i++) { net->W_1_grad[i] /= (double)net->batch_size; }    // W_1
	for (int i = 0; i < LAYER_1_NEURONS; i++) { net->b_1_grad[i] /= (double)net->batch_size; } // b_1
}



/*
	This function computes the gradient of the loss w.r.t. the weight matrix 
	of the output later, this is the first function to be called in the backprop()
	master function.

	backprop_W_2() retruns the gradient of lsos w.r.t. the pre activations of the 
	hidden layer, z_2. dL/dz_2 is need for the the subsequent function calls in the
	backprop() master function. dL_dz_2 is also the equivalent to the grad of J() 
	w.r.t. b_2

	The following mathematical computations are performed in this function:

		dL/dz_i = p_i - y_i

		dL/dWij = (dL/dz_i) * (dz_i/dW_ij)        
	            = (p_i - y_i) [outer propduct] h_j      
	
		Written equivalently as:

		dL/dW_2 = dL/dz_2 [outer product] hidden
		dL/db_2 = dL/dz_2

	to understand the derivation of these formulas, check the ReadME
*/
double* accumulate_W_2_grad(network* net, outputs* batch, int batch_element)
{
	/*
		----- A Note on the Shapes Involved in the below Matrix Operations: ------

		This funciton accumulates the gradient of the cost w.r.t. the weights of the
		last weight matrix of the model. However, this is done for each model output
		of the batch. Each gradient is computes then averaged over the batch length.


		Below, the following operation is performed:

		compute dL/dW_2 = dL/dz_2 [outer product] hidden

		Where:   W_2 has shape [10, 128]
				 net->hidden[batch example] has shape [128, 1]

				  Therefore the output product of dL/dz_2 by each
				  hidden state will be: [10, 128]

				  Which is the same shape as W_2. In the below algorithm,
				  each gradient is accumulated into the double* outer_prod
				  array we have allocated memory for. Each outer product is
				  accumulated into net->W_2_grad. After all gradients have
				  been accuumulated for the batch, each partial derivative
				  is averaged across the batch, to which it is readyu to use
				  for stochastic gradient descent.
	*/

	/*
		Step 1.) compute dL/dz_i = p_i - y_i 
		
			Where y_i is 1 if class i == 1

			And p_i is the models predicted probability 
			that the current example is of class i
	*/

	double* dL_dz_2 = malloc(10 * 1 * sizeof(double));  // allocate memory for dL_dz_2
	if (dL_dz_2 == NULL) { exit(1); }
	for (int i = 0; i < 10; i++) { dL_dz_2[i] = 0;}

	int y_i = 0;  

	for (int i = 0; i < 10; i++) 
	{
		if (batch[batch_element].target == i) {  // compute y_i for the current example of the batch
			y_i = 1;
		}
		else { y_i = 0; }

		dL_dz_2[i] = batch[batch_element].output_vector[i] - y_i;  // compute dL_dz_2
	}

	/*
		Step 2.) compute dL/dW_2 = dL/dz_2 [outer product] hidden
	*/
	
	double* dL_dW_2 = malloc(10 * 128 * sizeof(double));
	if (dL_dW_2 == NULL) { exit(1); }
	for (int i = 0; i < 10; i++) { dL_dW_2[i] = 0; }

	outer_product(dL_dz_2, net->hidden[batch_element], dL_dW_2, 10, 128);  // compute  dL/dz_2 [outer product] hidden

	for (int row = 0; row < 10; row++) {
		for (int col = 0; col < 128; col++) {
			net->W_2_grad[INDEX(row, col, 128)] += dL_dW_2[INDEX(row, col, 128)];  // accumulate gradient into W_2_grad inside the network struct for the current example.
		                                                                           // net.W_2_grad is averaged over the batch in the backprop() master function 
		}
	}

	free(dL_dW_2);  // temp dL_dW_2 no longer needed

	return dL_dz_2; // this is returned in order to propagate the grad backwards
					// this is also the grad of J() w.r.t. b_2
}

/*
	This function accumulates the gradient of the cost J() w.r.t.
	the bias vector the the output layer. This grad has already
	been computed within backprop_W_2(), so all that must happed
	is to accumulated into net.b_2_grad.

	dL/db_2 = dL/dz_2 <--- whihc must be passed in as an arg
*/
void accumulate_b_2_grad(network* net, double* dL_dz_2)
{
	for (int i = 0; i < 10; i++){
		net->b_2_grad[i] = dL_dz_2[i];  // accumulate grad
	}
}

/*
	This function propagates the gradient of the cost w.r.t. the weight matrix
	of the hiddne layer for a single example wihtin a batch. It takes as input 
	dL/dz_2, which was computed in accumulate_W_2_grad. 

	The func outputs dL_dz_1 which is used within accumulate_b_1_grad

*/
double* accumulate_W_1_grad(network* net, outputs* batch, double* dL_dz_2, int batch_element)
{


	/*  
		Step 1.)  compute W_2_Transpose

				W_2 is of shape [10, 128]

				Therefore: W_2_Transpose is of shape [128, 10]
	*/

	double* W_2_T = malloc(128 * 10 * sizeof(double));  
	if (W_2_T == NULL) { exit(1); }
	   
	for (int row = 0; row < 128; row++){              // initialize all val in W_2_T to 0
		for (int col = 0; col < 10; col++) {
			W_2_T[INDEX(row, col, 10)] = 0;
		}
	}

	transpose(W_2_T, net->W_2, 10, 128);  // perform transpose operation

	/*
		Step 2.) Compute dL/dhidden = W_2.Transpose * dL/dz_2

				W_2_T is of shape [128, 10]

				dL/dz_2 is of shape [10, 1]   (grad of loss w.r.t. pre activations of output layer)


				  W_2_T   * dL/dz_2  = dL_dhidden 
				[128, 10] * [10, 1]  = [128, 1]


	*/

	double* dL_dhidden = malloc(128 * 1 * sizeof(double));
	if (dL_dhidden == NULL) { exit(1); }

	for (int i = 0; i < 128; i++) {   // initialize to 0
		dL_dhidden[i] = 0; 
	}  

	matmul(dL_dhidden, W_2_T, dL_dz_2, 128, 10, 10, 1);   // perform matmul:  W_2_T * dL/dz_2  = dL_dhidden 

	free(W_2_T);  // W_2_T is also no longer needed

	/*
		Step 3.) compute ReLU'(z_1) 

			Where z_1 are the pre activations of the hidden layer  [128, 1]

			ReLU'(x) is the derivative of ReLU(x)

			ReLU(x) = max(0, x)

			ReLU'(x) = 1 for x >= 0
			         = 0 for x < 0

		This is computed for each z_1 from the batchj
	
	*/

	double* ReLU_prime_z_1 = malloc(128 * 1 * sizeof(double));   // allocate mem for ReLU'(z_1) 
	if (ReLU_prime_z_1 == NULL) { exit(1); }

	for (int i = 0; i < 128; i++) {  // initialize w/ zero
		ReLU_prime_z_1[i] = 0;
	}

	for (int i = 0; i < 128; i++) {  // compute piecewise derivative of ReLU()
		if (net->pre_activations_1[batch_element][i] >= 0) {
			ReLU_prime_z_1[i] = 1;
		}
		else {
			ReLU_prime_z_1[i] = 0;
		}
	}

	/*
		Step 4.) dL/dz_1 = dL/dhidden [elementwise multiplication] ReLU'(z_1)

			Where dL/dz_1 is the derivative of the loss w.r.t. the preactivationes of the hidden layer

			All vectors in this equation have a dimmensionality of 128.
	*/
	
	double* dL_dz_1 = malloc(128 * 1 * sizeof(double));
	if (dL_dz_1 == NULL) { exit(1); }

	for (int i = 0; i < 128; i++) {  // initialize to 0
		dL_dz_1[i] = 0; 
	}

	for (int i = 0; i < 128; i++) {
		dL_dz_1[i] = dL_dhidden[i] * ReLU_prime_z_1[i];
	}

	free(dL_dhidden);     // these no longer needed
	free(ReLU_prime_z_1); 

	/*
		Step 5.) compute dL/dW_1 = dL/dz_1 [outer product] input
	*/

	double* dL_dW_1 = malloc(784 * 1 * sizeof(double));  // allocate mem for dL_dW_1
	if (dL_dW_1 == NULL) { exit(1); }

	for (int i = 0; i < 784; i++) {
		dL_dW_1[i] = 0;
	}

	outer_product(dL_dz_1, batch[batch_element].input_vector, dL_dW_1, 128, 784);  // op shape --> [128, 784]
	

	/*
		Step 6.)  acuumulate dL/dW_1 in the network struct, then average across batch size.

				dL/db_1 = dL/dz_1 will be reteurn and accumulated into the input layer bias 
				parameters inside of the accumulate_b_1_grad() function.
	*/

	for (int row = 0; row < 128; row++) {
		for (int col = 0; col < 784; col++) {
			net->W_1_grad[INDEX(row, col, 784)] += dL_dW_1[INDEX(row, col, 784)];  // accumulate grad into network struct
		}
	}

	free(dL_dW_1);

	return dL_dz_1;
}


/*
	This function accumulates the gradient w.r.t. the bias parameters of the 
	input layer. The grad has already been precomputed in accumulate_W_1_grad,

		dL_db_i = dL_dz_1
*/
void accumulate_b_1_grad(network* net, outputs* batch, double* dL_dz_1, int batch_element) {

	for (int i = 0; i < 128; i++) {

		net->b_1_grad[i] += dL_dz_1[i];  // accumulate precomputed grad into network struct
	}
}









