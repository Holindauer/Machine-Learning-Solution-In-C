#pragma once

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