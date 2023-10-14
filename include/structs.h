#pragma once

#include "libraries.h"


/*
    This struct is used to hold the examples and labels for the dataset.

    examples is a type double array of arrays. The nested arrays within
    .examples are the matricies indexable with the INDEX macro.

    labels is an array containing the labels as type double
*/ 

typedef struct {
    double** examples;
    double* labels;
}dataset;


/*
    This struct is used to hold the weight matricies of a multi layer 
    perceptron as a double array. Along with each matrix' shape stored 
    as individual integers.

    It is intended to be used initialize with the create_matrix() function
*/
typedef struct {

    double* W_1;             // layer 1 weight matrix
    int W_1_rows, W_1_cols;  // layer 1 shape


    double* W_2;             // layer 2 weight matrix
    int W_2_rows, W_2_cols;  // layer 1 shape
    

    double* W_3;             // layer 3 weight matrix
    int W_3_rows, W_3_cols;  // layer 1 shape

}network_weights;