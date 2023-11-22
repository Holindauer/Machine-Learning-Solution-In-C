#pragma once

#include "structs.h"
#include "libraries.h"

//load_data.c
void init_dataset(Data* dataset);
void print_dataset(Data dataset);
void load_data(Data* dataset, FILE* stream);


//model.c
void init_model(Model* model);


//matrix.c
void MatMul(double* dest_matrix, double* src_A, double* src_B, int A_rows, int A_cols, int B_cols);
void Display_Matrix(double* matrix, int rows, int cols);