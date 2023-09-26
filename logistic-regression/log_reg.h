#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <math.h>

// struct to hold individual examples within an array.
typedef struct {
	int X;			// integer feature
	int y;			// integer target (binary divisibility status)
}example;

// struct to hold individual model predictions within an array.
typedef struct {
	double probability;    // sigmoid/logistic output
	int prediction;		   // probability post threshold function
}pred;


// struct for return multiple gradients from the compute_gradient() fucntion
typedef struct {
	double dJ_dw;    // partial derivative for weight term
	double dJ_db;    // partial derivative for bias term
}gradient;




void populate_data_split(example data_split[], int num_examples);

double logistic_function(int X, double w, double b);

int pred_threshold(double probability, double threshold);

double log_loss(pred predictions[], example data_split[], int num_examples);

gradient compute_gradient(pred predictions[], example data_split[], int num_examples);






