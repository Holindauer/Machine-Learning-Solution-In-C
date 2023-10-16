#pragma once

#include "structs.h"

//--------------------------------------------------------------------------------------weight_initialization.c
double random_double(double min, double max);

void he_initialize(double* weight_matrix, int rows, int cols);

//--------------------------------------------------------------------------------------matrix_operations.c

double* create_matrix(int rows, int cols);

void display_matrix(double* matrix, int rows, int cols);

void matmul(double* C, double* A, double* B, int rows_A, int cols_A, int rows_B, int cols_B);

//--------------------------------------------------------------------------------------activation_functions.c


void ReLU(double* matrix, int rows, int cols);

void Softmax(double* vector, int length);


//--------------------------------------------------------------------------------------model.c

/*
    Runs a forward pass using network weights, a single example, 
    and places result into model_output array.
*/
void forward(weights net, double* example, double* model_output);


/*
    Computes the argmax of the model model output probability vector
*/
int predict(double* model_output);

//--------------------------------------------------------------------------------------memory.c
void check_memory_allocation(double* arr);

void free_dataset(example* dataset, int num_examples);

//--------------------------------------------------------------------------------------load_data.c
void load_data(const char* filename, example* dataset, int num_digits);