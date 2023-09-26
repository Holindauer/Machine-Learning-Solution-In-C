#include "log_reg.h"

void populate_data_split(example data_split[], int num_examples) {

	// loop throught example array and pass binary divisibility status of iteration into targets
	// a divisibility status of 1 means the integer in question is divisible by 2. We are modifying 
	// an array in place so there is no need to return

	for (int i = 0; i <= (num_examples - 1); i++) {
		// if divisible by 2:
		if (i % 2 == 0) {
			data_split[i].X = i;   // set feature to iter
			data_split[i].y = 1;   // set div status to 1
		}
		// otherwise:
		else {
			data_split[i].X = i;   // set feature to iter
			data_split[i].y = 0;   // set div status to 0 
		}
	}
}

double logistic_function(int X, double w, double b) {

	// this funciton returns a probability from the logistic function
	// parametrized by a weight an bias in the z term.  sigmoid(z) = 1 / (1 + exp(-z))

	// initialize z term
	double z = 0, probability = 0;

	// perform  weighted sum
	z = (w * X) + b;

	// compute logits of logistic function
	probability = 1 / (1 + exp(-z));

	return probability;

}

int pred_threshold(double probability, double threshold) {
	// this function takes a single model prediction from the logistic
	// function and assigns it a binary prediction determined by if it 
	// exceeds the prediction threshold.

	// initialize pred to 0
	int local_pred = 0;

	// check if model output exceeds the threshold
	if (probability >= threshold) {
		local_pred = 1;	   // assign 1 if so
	}

	return local_pred;
}

double log_loss(pred predictions[], example data_split[], int num_examples) {
	
	// This function computes the log loss of the epoch
	// from an array of preds and an array of examples
	// 
	// L = - (1/N) * Sum(i=1 to N) [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
	// where N is num_examples, y_i is the ith target, p_i is the ith probability

	double extracted_probability = 0;
	int extracted_target = 0;

	double loss = 0;

	// set epsilon value for preventing underflow error
	double epsilon = 1e-15;

	// begin summation of loss
	for (int i = 0; i < num_examples; i++) {

		// extract targets and preds from array and cast to double
		extracted_probability =  predictions[i].probability;

		// Ensure that the extracted prob is within the bounds (epsilon, 1-epsilon)
		extracted_probability = fmax(fmin(1 - epsilon, extracted_probability), epsilon);

		extracted_target = data_split[i].y;

		// add summation argument to running summation for example
		loss += ( extracted_target * log(extracted_probability) ) + ( (1 - extracted_target) * log(1 - extracted_probability) );
	}

	// average loss across num examples. Then make negative
	loss = -(loss / num_examples);

	return loss;
}

gradient compute_gradient(pred predictions[], example data_split[], int num_examples) {
	/* The partial deriv of the log loss funciton with respect to each param is
	   given by: 

	   For Weight Terms
	   dJ/d_weight_i = (1 / N) sigma_N_i=1 (p_i - y_i) Xij

	   For Bias Terms:
	   dJ/d_bias_i = (1 / N) sigma_N_i=1 (p_i - y_i)

	   where N is the num examples, 
	         p_i is the i'th probability,
			 y_i is the i'th target,
			 Xij is the i'th X value of the j'th parameter 

     */

	// initialize gradient for w and b
	double dJ_dw = 0, dJ_db = 0;

	// initialize extracted probabilities
	double extracted_probability = 0;
	
	// initialize extracted target and its X value
	int extracted_target = 0, extracted_X = 0;

	// accumulate gradient across examples
	for (int i = 0; i < num_examples; i++) {
		
		// extract probabilities
		extracted_probability = predictions[i].probability;

		// extract predictions
		extracted_target = data_split[i].y;

		// extract X value for the given prediction
		extracted_X = data_split[i].X;

		// accumulate gradient for weight term
		dJ_dw += (extracted_probability - extracted_target) * extracted_X;

		// accumulate gradient for bias term
		dJ_db += (extracted_probability - extracted_target);

	}

	// average gradient accumulation across num_examples
	dJ_dw /= num_examples;
	dJ_db /= num_examples;

	// initialize gradient struct for the return of the gradients
	gradient grad_return;

	// populate grad_return with gradients for the epoch
	grad_return.dJ_dw = dJ_dw;
	grad_return.dJ_db = dJ_db;

	return grad_return;
}