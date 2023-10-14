#include "libraries.h"
#include "structs.h"
#include "functions.h"




/*  
    load_data() loads in mnist digits from a csv file where each row of the csv is 
    a flattened image. The first column is the label of the image and the subsequent 
    columns correspond to the flattened pixel intensities. 

    load_data() takes two arguments. The first is const char* filename which is a 
    string containing the name of the csv file holding the data. The second arg is
    int num_digits, which is the number of examples that will be extracted from the 
    dataset.
*/
dataset load_data(const char* filename, int num_digits)
{
    //--------------------------------------------------------------------------------------------------------------------Open File
    FILE* file = fopen(filename, "r");                                  // open file input stream
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file %s\n", filename);          // error handling
        exit(1);
    }

    //--------------------------------------------------------------------------------------------------------------------Initialize pointer array for examples 
    //                                                                                                                    and regular array for labels


    dataset data;                                                      // initialize a dataset struct to hold

    data.examples = (double**)malloc(num_digits * sizeof(double*));    // allocate memory for examples
    data.labels = (double*)malloc(num_digits * sizeof(double));        // allocate memory for labels

    if (data.examples == NULL || data.labels == NULL) {                // handle memory allocation error
        fprintf(stderr, "Failed to allocate memory\n");
        fclose(file);
        free(data.examples);  // Note: It's safe to free a NULL pointer
        free(data.labels);    // So these are okay even if the memory wasn't allocated
        exit(1);
    }

    double label = 0;                                                   // initialize a double to store an 
                                                                        // examples label
 
    double pixel_intensity = 0;                                         // initialize an integer to hold each  
                                                                        // individual pixel intensities values extracted

    //--------------------------------------------------------------------------------------------------------------------Read in Data


    for (int example = 0; example < num_digits; example++)            // loop through the specified number of digits to extract from the dataset
    {

        double* digit = (double*)malloc(784 * sizeof(double));        // Allocate memory for each digit at each iteration of the outer loop
        if (digit == NULL) {
            fprintf(stderr, "Failed to allocate memory for digit\n");
            fclose(file);
            free(data.labels);
            // Free any previously allocated digit memory
            for (int i = 0; i < example; i++) {
                free(data.examples[i]);
            }
            free(data.examples);
            exit(1);
        }


        fscanf(file, "%lf,", &label);       // scan in label of the current example
        data.labels[example] = label;       // place label into the example'th label of the dataset


        for (int pixel = 0; pixel < 784; pixel++)                // iterate through the pixel values of each digit within the csv
                                                                     
        {
            fscanf(file, "%lf", &pixel_intensity);               // scan in pixel intensity for the pixel'th pixel

            digit[pixel] = pixel_intensity;                      // place pixel intensity into the pixel'th pixel of digit

        }

        data.examples[example] = digit;     // place digit pointer into example'th example
    }

 
    fclose(file);     

    return data;
}

/*
    This function frees the memory allocated
    for the dataset after done with it
*/
void free_data(dataset* data, int num_examples) {
    // Free each digit
    for (int i = 0; i < num_examples; i++) {
        free(data->examples[i]);
    }
    // Free the examples and labels arrays
    free(data->examples);
    free(data->labels);
}
