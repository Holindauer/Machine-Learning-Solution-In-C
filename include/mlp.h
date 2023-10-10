#pragma once


float random_float(float min, float max);

void he_initialize(float* matrix, int rows, int cols);

void ReLU(float* matrix, int rows, int cols);

/*
    This function defines the forward pass for a 3 layer mlp.
    Provided weight matrices, num neurons per layer, input and
    its associated shape. The function outputs a matrix pointer
    array.
*/
double* forward_pass(double* W_1, double* W_2, double* W_3,     // weight matricies
    int layer_1_nodes, int layer_2_nodes, int layer_3_nodes,    // num neurons per layer
    double* X, int X_rows, int X_cols);                         // input and input dimmensions