#pragma once

#include "libraries.h"



// this struct is used to hold a pointer array to a pointer array of flattened mnist digit matricies
// and a pointer array of those digits associated labels
typedef struct {
    double** examples;
    double* labels;
}dataset;

// This struct is used to construct a batch of examples from the larger dataset that exists within the 
// a dataset struct. A batch struct type will be used for storing both incoming batches to the model for
// inferrence as well as predictions that will be fed into the loss function.
// predictions contains all probabilities output by the model that an image belongs to each individual 
// class
typedef struct {
    double** examples;
    double* labels;
    double** predictions;
    int batch_size;
}batch;



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


/*
    This function will be used to seperate out a number of batches from the larger dataset, 
    stored within a dataset struct. Provided a start and end index of the dataset, a batch 
    will be construct with examples from between those indices.

*/
batch gather_batch(dataset data, int batch_size, int start_batch);


