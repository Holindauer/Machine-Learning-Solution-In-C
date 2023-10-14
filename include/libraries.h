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


// included libraries
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

