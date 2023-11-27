#pragma once

#include "structs.h"
#include "libraries.h"

//load_data.c
void init_dataset(Data* dataset);
void print_dataset(Data dataset);
void load_data(Data* dataset, FILE* stream);

//model.c
void init_model(Model* model);
double forward(Model* model, double example[NUM_FEATURES], int example_number);

//matrix.c
void MatMul(double* dest_matrix, double* src_A, double* src_B, int A_rows, int A_cols, int B_cols);
void Display_Matrix(double* matrix, int rows, int cols);
void Elementwise_Addition(double* destination, double* matrix_to_add, int rows, int cols);
void Elementwise_Subtraction(double* destination, double* matrix_to_subtract, int rows, int cols);
void Elementwise_Multiply(double* destination, double* matrix1, double* matrix2, int rows, int cols);
void Copy_Matrix(double* destination, double* matrix_to_copy, int rows, int cols);
void Transpose_Matrix(double* destination_matrix, double* src_matrix, int src_rows, int src_cols);
void Zero_Matrix(double* matrix, int rows, int cols);

// utils.c
int argmax_vector(double* vector, int vector_length);
double rand_double();

// activation_functions.c
void Elementwise_Softmax(double* matrix, int vector_length);
Elementwise_ReLU(double* matrix, int rows, int cols);

// loss_function.c
double Categorical_Cross_Entropy(double output[][LAYER_2_NEURONS], double targets[][NUM_CLASSES]);

// metrics.c
double Accuracy(double prediction[LAYER_2_NEURONS], double targets[][NUM_CLASSES]);

// backprop.c
void Elementwise_Subtraction(double* destination, double* matrix_to_add, int rows, int cols);
void backprop(Model* model, Data* dataset, int example_num);
void ReLU_Prime(double* matrix_to_differentiate, int rows, int cols);

// sgd.c
void zero_init_gradient_accumulator(Gradient* gradient_accumulator);
void Accumulate_Gradient(Model* model, Gradient* gradient_accumulator);
void Stochastic_Gradient_Descent(Model* model, Gradient* gradient_accumulator, double learning_rate);