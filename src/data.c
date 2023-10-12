#include "libraries.h"


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


dataset load_mnist_digits(const char* filename, int num_digits)
{

    FILE* file = fopen(filename, "r");                                    // open file input stream
    if (file == NULL)       
    {
        fprintf(stderr, "Could not open file %s\n", filename);            // error handling
        exit(1);
    }

    double** digits = (double**)malloc(num_digits * sizeof(double*));     // Allocate memory for an array of pointers to 
    if (digits == NULL)                                                   // pointers (matricies from define_new_matrix())
    {
        fprintf(stderr, "Failed to allocate memory\n");                   // error handling
        fclose(file);
        exit(1);
    }

    double* labels = define_new_matrix(num_digits, 1);                    // allocate memory for a labels matrix

    for (int d = 0; d < num_digits; d++)
    {
        digits[d] = define_new_matrix(784, 1);            // at the d'th index of digits, define a new matrix

        for (int i = 0; i < 784; i++)                     // iterate through the pixel values of each digit within the csv
        { 
            if (i == 0)                                   // if first number in csv, that number is the image label
            {
                fscanf(file, "%lf,", &labels[d]);         // scan in label for d'th digit
            }

            if (i < 783)                                  // scan digits until last pixel val
            {   
                fscanf(file, "%lf,", &digits[d][INDEX(i, 1, 1)]);       // into the d'th example of digits, place the i'th pixel intensity scanned in from  
            }                                                           // the csv into the newly defined matrix at the corrext index using the INDEX macro
                                                                        // the comma after %lf will only work until the last line, where there will be a \n

            else // Last element in a line  
            {
                fscanf(file, "%lf\n", &digits[d][i]);                   // for the last line of a csv row, handle the newline case mentioned in the above comment
            }
        }
    }

    dataset mnist_dataset;            // initialize a dataset struct for return statement

    mnist_dataset.examples = digits;  // place examples into dataset
    mnist_dataset.labels = labels;    // place labels into dataset



    // close file
    fclose(file);   
            
    return mnist_dataset;
}



/*
   This function returns a batch of batch_size from a dataset struct
*/
batch gather_batch(dataset data, int batch_size, int start_batch)
{
    batch new_batch;                    // initialzie new batch
     
    new_batch.batch_size = batch_size;  // set batch size


    for (int i=0; i < batch_size; i++)
    {
        // place start of batch + i'th example into the batch
        new_batch.examples[start_batch + i] = data.examples[start_batch + i];

        // place start of batch + i'th label into the batch
        new_batch.labels[start_batch + i] = data.labels[start_batch + i];

    }

    return new_batch;
}


