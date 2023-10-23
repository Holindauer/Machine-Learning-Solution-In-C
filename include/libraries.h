#pragma once

#define _CRT_SECURE_NO_WARNINGS

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define W_1_ELEMENTS 128 * 784  // weight matrix elements
#define W_2_ELEMENTS 10 * 128

#define LAYER_1_NEURONS 128
#define LAYER_2_NEURONS 10

#define INDEX(i, j, cols) ((i) * (cols) + (j)) // for indexing a flattened matrix arr

#define INPUT_FEATURES 784  // mnist specific


// included libraries
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>