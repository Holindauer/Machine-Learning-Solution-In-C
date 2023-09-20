#include "linear-regression.h"


void get_target_function(double* w, double* b) {
	// this functions obtains the parameters of the target function 
	// that this programm will approximate with gradient descent

	// prompt user
	printf("\n\nEnter the slope and y intercept of the target model to approximate: ");

	// get user input
	printf("\n\n slope: "), scanf("%lf", w), printf("\n y intercept: "), scanf("%lf", b);

	// display result
	printf("\nf(x) = %lfx + %lf", *w, *b);

}

double linear_function(double x, double w, double b) {
	// this function computes the y value for a linear function
	double y = (x * w) + b;
	return y;
}

int random_x(void) {
	// this function computes a random number in range [-100, 100]

	// setbounds of random x
	int a = -100, b = 100;

	// standard rand range formula
	int random_x = a + rand() % (b - a + 1);

	return random_x;
}

array_return predict(double w_theta, double b_theta, xy_coord data_split[], int num_examples) {
	// this function returns a struct containing an array of predictions for a given linear model

	// intialize variable to store current examples X value in
	int X_example = 0;

	// intialize var to store the currect precition
	double y_pred = 0;

	// intialize an array_return struct to return predictions as an array
	array_return y_hat;

	// loop through example X values and predict y_hat
	for (int i = 1; i <= num_examples; i++) {

		// get current example x val
		X_example = data_split[i - 1].x;

		// predict on example
		y_pred = linear_function(X_example, w_theta, b_theta);

		// populate y_hat's array (with the number of train examples) with the current pred
		y_hat.train_arr[i - 1] = y_pred;
	}

	return y_hat;
}


double MSE(double predictions[], double targets[], int num_examples) {
	// This function computes the mean squared error from of the current
	// model from an array of predictions and of targets and num examples

	// J(X) = (1 / 2m) m_summation_i=1( ( h(x_i) - y_i))^2 )

	// intialize mse
	double mse = 0;

	for (int i = 1; i <= num_examples; i++) {
		// square of the difference of pred and targ
		mse += pow((predictions[i - 1] - targets[i - 1]), 2);
	}

	// compute 1/2 the average of the squared error
	mse = mse / num_examples / 2;

	return mse;
}


gradient gradient_computation(double predictions[], double targets[], int num_examples, xy_coord train[], gradient computed_grad) {
	/*
		This function computes the gradient of the cost with
		respect to w and b. The equations are below

		J(X) = (1 / 2m) m_summation_i=1 ((h(x_i) - y_i)) ^ 2 )    ---Loss

		dJ/dw = (1/m) m_summation_i=1 ( h(x_i) - y_i ) x_i        ---Partial Derivative wrt w
		dJ/db = (1/m) m_summation_i=1 ( h(x_i) - y_i )            ---Partial Derivative wrt b

		Where m = num_examples
	*/

	// Initialize partial derivatives of the cost 
	double dJ_dw = 0;
	double dJ_db = 0;

	// accumulate gradients across examples
	for (int i = 1; i <= num_examples; i++) {
		dJ_dw += (predictions[i - 1] - targets[i - 1]) * train[i - 1].x; // w partial derivative
		dJ_db += (predictions[i - 1] - targets[i - 1]);                  // b partial derivative
	}

	// average grads across examples
	double num_examples_double = num_examples;  // avoid integer division issues
	dJ_dw = dJ_dw / num_examples_double;
	dJ_db = dJ_db / num_examples_double;

	// package grads into gradient struct
	computed_grad.w_update = dJ_dw;
	computed_grad.b_update = dJ_db;

	return computed_grad;
}

gradient epoch(double w_theta, double b_theta, double train_pred[], xy_coord train[], double train_targets[], gradient grad) {
	// This function computes the gradient of the cost for a single epoch. Performing predictions, computing MSE and grad
	// The gradient is returned in a gradient struct that contains two doubles that will be used in the GD update rule

	// get array_return struct from predict() call
	array_return y_hat = predict(w_theta, b_theta, train, 100, train_pred);

	// extract the predictions
	for (int i = 1; i <= 100; i++) {
		// put prediction in its example index
		train_pred[i - 1] = y_hat.train_arr[i - 1];

		// get example X and y
		int X_targ = train[i - 1].x;
		double y_targ = train[i - 1].y;

		// put target into target array
		train_targets[i - 1] = y_targ;

		// print out for debugging
		//printf("\n Extracted Prediction for example %d: y_hat = %lf   ---  Target: (%d, %lf)", i, train_pred[i - 1], X_targ, y_targ);
	}

	// compute loss for current epoch with 100 example training set
	double loss = MSE(train_pred, train_targets, 100);


	// compute gradient of cost
	grad = gradient_computation(train_pred, train_targets, 100, train, grad);

	// Display performance for current epoch
	printf("\nLoss : % lf - - - Gradient: dJ/dw = %lf dJ/db = %lf\n", loss, grad.w_update, grad.b_update);

	return grad;
}