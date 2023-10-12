#pragma once

#include "libraries.h"


double random_double(double min, double max);

void he_initialize(double* matrix, int rows, int cols);

void ReLU(double* matrix, int rows, int cols);

/*
    This function defines the forward pass for a 3 layer mlp.
    Provided weight matrices, num neurons per layer, input and
    its associated shape. The function outputs a matrix pointer
    array.
*/
double* forward_pass(double* W_1, double* W_2, double* W_3,     // weight matricies
    int layer_1_nodes, int layer_2_nodes, int layer_3_nodes,    // num neurons per layer
    double* X, int X_rows, int X_cols);                         // input and input dimmensions


/*
    This function computes the cross entropy loss for a batch during training.
    The batch struct contains both the example matrix, label array, and predictions
    array within its members.

    cross entropy for a single datapoint is defined below as:

    L(y, y_hat) = - M_sigma_c=1 (y_c * log(y_hat_c) )

    Where:  M is the number of classes

            M_sigma_c=1 is the summation starting at 1 to the num classes

            y_c is a binary value indicating whether the target label and the
            c'th label for the current example are the same.

            y_hat_c is the the model's predicted probability of whether the
            current example belongs to class c

    Thus, cross entropy for the entire batch is:

    batch_loss = -(1/N) N_sigma_i=1 ( L(y, y_hat) )

    Where N is the number of examples in the batch
*/
double cross_entropy_loss(batch train_batch, int num_classes);

void Softmax(double* matrix, int rows, int cols);
