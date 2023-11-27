#include "libraries.h"
#include "structs.h"
#include "functions.h"


/*
This function performs backpropagation for the gradient of categorical cross entropy with respect to 
all states throughout the backward pass. Gradients are computed directly into the model struct. This 
means that this function must be applied for every
 
The following are the partial derivatives of every state in the network.
Lowercase are vectors, uppercase are matrixies.
 
dL/dOutput = output - y

dL/logits = output - y

dL/dW_2 = dL/dlogits * dlogits/dW_2 = (output - y) * hidden.T
dL/dbias_2 = output - y

g = ReLU'(hidden)
dL/dhidden = W_2.T * (output - y) [outer prod] g

dL/dW_1 = dL/dhidden * dhidden/dW_1 = dL/dhidden * input.T
dL/dbias_1 = dL/dhidden
*/

void backprop(Model *model, Data* dataset, int example_num) {

	// NOTE: model->logits_Grad is copied multiple times because it contains output - y


	// dL / dOutput = output - y
	Copy_Matrix(model->output_Grad, model->output[example_num], NUM_CLASSES, 1);
	Elementwise_Subtraction(model->output_Grad, dataset->features[example_num], NUM_CLASSES, 1);

	// dL / dOutput = output - y
	Copy_Matrix(model->logits_Grad, model->output_Grad, NUM_CLASSES, 1);

	// dL/dW_2 = dL/dlogits * dlogits/dW_2 = (output - y) * hidden.T
	// NOTE: because hidden is a vector, ransposing is as simple as reversing the rows/cols on input
	MatMul(model->W_2_Grad, model->logits_Grad, model->hidden[example_num], LAYER_2_NEURONS, LAYER_1_NEURONS, LAYER_1_NEURONS);  // <-- check

	// dL/dbias_2 = output - y
	Copy_Matrix(model->b_2_Grad, model->logits_Grad, LAYER_2_NEURONS, 1);

	// g = ReLU'(hidden)
	// dL / dhidden = W_2.T * (output - y)[outer prod] g
	
	double hidden_Temp[LAYER_1_NEURONS] = { 0 };
	
	// First calculate W_2.T * (output - y)
	double W_2_Transpose[LAYER_2_NEURONS * LAYER_1_NEURONS] = { 0 };
	Transpose_Matrix(W_2_Transpose, model->W_2, LAYER_2_NEURONS, LAYER_1_NEURONS);

	MatMul(hidden_Temp, W_2_Transpose, model->logits_Grad, LAYER_2_NEURONS, LAYER_1_NEURONS, 1);

	// Then element-wise multiply with g (ReLU derivative)
	double g[LAYER_1_NEURONS] = { 0 };
	Copy_Matrix(g, model->hidden[example_num], LAYER_1_NEURONS, 1);
	ReLU_Prime(g, LAYER_1_NEURONS, 1);

	Elementwise_Multiply(model->hidden_Grad, hidden_Temp, g, LAYER_1_NEURONS, 1);

	// dL/dW_1 = dL/dhidden * dhidden/dW_1 = dL/dhidden * input.T
	
	double input_Transpose[NUM_FEATURES * LAYER_1_NEURONS] = { 0 };
	Transpose_Matrix(input_Transpose, model->input[example_num], NUM_FEATURES, 1);

	MatMul(model->W_1_Grad, model->hidden_Grad, input_Transpose, LAYER_1_NEURONS, 1, NUM_FEATURES);

	
	// dL / dbias_1 = dL / dhidden
	Copy_Matrix(model->b_1_Grad, model->hidden_Grad, LAYER_1_NEURONS, 1);



}

// this funciton implements the derivative of ReLU on a contiguous 2D array in place
void ReLU_Prime(double* matrix_to_differentiate, int rows, int cols) {

	for (int i = 0; i < (rows * cols); i++) {
		if (matrix_to_differentiate[i] < 0) {
			matrix_to_differentiate[i] = 0;
		}
	}

}