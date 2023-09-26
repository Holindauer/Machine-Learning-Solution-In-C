/*
	This program implements gradient descent for logistic regression to
	train the logistic function to approximate the divisibility by two
	logical function. Which I have written below in python:

	if input % 2 == 0:
		divisibility = 1
	elif input % 2 != 0:
		divisibility = 0

*/

#include "log_reg.h"


int main(void) {

	//---------------------------------------------------------------------------------------------< 1. Get Training Data >

	// instantiate an array of example structs to hold training data
	example train[100];

	// set num examples
	int num_examples = 100;

	// populate training set with 100 examples. train array modified in place
	populate_data_split(train, num_examples);


	//---------------------------------------------------------------------------------------------< 2. Initialize Model >

	// initialize model weights
	double w = 1, b = 1; 

	// intialize threshold for pred_threshold()
	double prediction_threshold = 0.5;

	// initialize a pred array to store double probabilities and int predictions 
	pred predictions[100];

	//---------------------------------------------------------------------------------------------< 3. Training Loop >

	// The model makes predictions in two parts. First the logistic_function() is called to 
	// produce a probability, Then a threshold function is applied to produce a binary prediction.
	double probability = 0;
	int prediction = 0;

	// intitaize number of epochs 
	int epochs = 10000;

	// intialize learnign rate
	double lr = 0.001;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		// initialize loss value
		double loss = 0;

		// iterate through length of training set
		for (int i = 0; i < num_examples; i++) {

			// compute certainty that X is divisble by 2
			probability = logistic_function(train[i].X, w, b);

			// make prediction
			prediction = pred_threshold(probability, prediction_threshold);

			// store probability and prediction in pred array
			predictions[i].probability = probability;
			predictions[i].prediction = prediction;

		}

		//compute loss
		loss = log_loss(predictions, train, num_examples);

		// Gradient Computaiton
		gradient grad = compute_gradient(predictions, train, num_examples);

		// perform learning update 
		w = w - (lr * grad.dJ_dw);
		b = b - (lr * grad.dJ_db);

		// log epoch progress
		printf("\n\nEpoch: %d dJ/dw = %lf --- dJ_db = %lf --- Loss: %lf", epoch, grad.dJ_dw, grad.dJ_db, loss);
	
	}

	return 0;
}