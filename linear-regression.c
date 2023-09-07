/*
	This program implements gradient descent for  linear in C
*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h> // for rand()
#include <time.h>
#include <math.h>


// This struct will be used to represent the (x, y) points that will
// be used to train the linear model. An array will be populated with them.
typedef struct {
	double x;
	double y;
}xy_coord;

// this struct will be used to return arrays from functions there are two arrays, one 
// with the number of examples for the train set and the other with the num examples 
// for the test set
typedef struct {
	double train_arr[100];
}array_return;

// this struct will be used to return the updated gradients
//  of the cost with respect to the parameter vecto
typedef struct {
	double w_update;
	double b_update;
}gradient;


//Function Definitions
void get_target_function(double* w, double* b);																						// get target linear function form user

double linear_function(double x, double w, double b);																				// linear function

array_return predict(double w_theta, double b_theta, xy_coord data_split[], int num_examples);										// predict on a data split

double MSE(double predictions[]);																								    // Mean Squared Error

gradient gradient_computation(double predictions[], double targets[], int num_examples, xy_coord train[], gradient computed_grad);	//compute grad

gradient epoch(double w_theta, double b_theta, double train_pred[], xy_coord train[], double train_targets[], gradient grad);		// run gradient descent for single epoch



int main(void) {

	srand(time(NULL));  // Seed the random number generator

	// --------------------------------STEP 1--------------------------------<Get Target Linear Function>
	// define from user input the target linear function that 
	// our randomly initialize model will be trained to approximation
	double w_true = 0, b_true = 0;

	get_target_function(&w_true, &b_true);


	// --------------------------------STEP 2--------------------------------<Get Training Datasets from the inputted Function>
	// This creates an array of 100 xy_coord structs
	// called train. It will be populated with examples 
	// from the target linear model for use as the training
	// and test datasets respectively
	xy_coord train[100];


	// These variables will be used to house the X and y coordinates generated by
	// The linear function in order to populate our trian and test arrays with them

	int X = 0;   
	double y = 0;
	
	printf("\n\nNow Populating the Train Set with Examples From f(x)...\n\n");

	// populate train set

	for (int i = 1; i <= 100; i++) {
		// get random x value
		X = random_x();

		// get linear target y value from the x val
		y = linear_function(X, w_true, b_true);

		// populate array with x and y vals
		train[i - 1].x = X;
		train[i - 1].y = y;
	}
	
	// --------------------------------STEP 3--------------------------------<Intialize the Hypothesis Function Weights>

	// initialize model parameters to use as our model
	double w_theta = 1,
		b_theta = 1;

	// --------------------------------STEP 3--------------------------------<Train Model>

	// Initialize array to house the training predictions
	double train_pred[100] = { 0 };

	// initialize array to hold targets 
	double train_targets[100] = { 0 };

	// intialize gradient struct for learning update
	gradient grad;
	grad.w_update = 0;
	grad.b_update = 0;

	// intializer num epochs and learning rate
	int epochs = 100;
	double learning_rate = 0.0001;

	// training loop
	for (int i = 1; i <= epochs; i++) {
		grad = epoch(w_theta, b_theta, train_pred, train, train_targets, grad);
		printf("\n\nEpoch %d ---- Gradient When Returned To Main: dJ/dw = %lf ---- dJ/db = %lf", i, grad.w_update, grad.b_update);

		// learning rule
		w_theta = w_theta - (learning_rate * grad.w_update);
		b_theta = b_theta - (learning_rate + grad.b_update);

	}

	// display final model
	printf("\n\n-----------RESULTS!!!-----------");
	printf("\n\n\nTarget Model: y = %lfx + %lf", w_true, b_true);
	printf("\n\nHypothesis: y = %lfx + %lf", w_theta, b_theta);
	printf("\n\n-----------RESULTS!!!-----------\n\n");

	return 0;
}

void get_target_function(double *w, double *b) {
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
		y_hat.train_arr[i -1] = y_pred;
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