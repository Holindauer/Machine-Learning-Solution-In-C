#pragma once

#include "libraries.h"



// this struct is used to hold a pointer array to a pointer array of flattened mnist digit matricies
// and a pointer array of those digits associated labels
typedef struct {
    double** examples;
    double* labels;
}dataset;

/*
    Overview:
    This function will load in mnist digits from a csv file. The specific format of mnist data
    this function accepts is once where each digits is flattened into a row of the csv file.



    Input Parameters:
    The function requires you input the number of digits you wish to extract from the file as
    well as the filename of the csv

    Output Specs:

    Preconditions:
    This function uses the matmul.h header file when it calls define_new_matrix()

*/
dataset load_mnist_digits(const char* filename, int num_digits);