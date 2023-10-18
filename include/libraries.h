#pragma once

#define _CRT_SECURE_NO_WARNINGS

// define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/*
	This macro is used to access the 1d index of a 2d
	matrix (represented abstractly in a 1d array).
*/
#define INDEX(i, j, cols) ((i) * (cols) + (j))

/*
	These macros outline the shape of each weight matrix 
	making up the model.
*/
#define W_1_ELEMENTS 128 * 784
#define W_2_ELEMENTS 10 * 128


/*
	This is the input shape of mnist digits
*/
#define INPUT_FEATURES 784

/*
	These macros define the number of neurons of at each 
	layer of the network.
*/
#define LAYER_1_NEURONS 128
#define LAYER_2_NEURONS 10




// included libraries
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
