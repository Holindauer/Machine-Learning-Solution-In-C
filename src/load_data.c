#include "libraries.h"
#include "functions.h"
#include "structs.h"

void load_data(const char* filename, example* dataset, int num_digits)
{
    //--------------------------------------------------------------------------------------------------------------------Open File
    FILE* file = fopen(filename, "r");                                  // open file input stream
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file %s\n", filename);          // error handling
        exit(1);
    }

    //--------------------------------------------------------------------------------------------------------------------Read in Data

    int label = 0;               // initialize an int to store labels
    double pixel_intensity = 0;  // initialize a double to hold each individual pixel intensity

    // loop through the specified number of digits to extract from the dataset
    for (int example = 0; example < num_digits; example++)             
    {

        fscanf(file, "%d,", &label);       // scan in label of the current example
        dataset[example].label = label;    // place label into the example'th label of the dataset

        for (int pixel = 0; pixel < 784; pixel++)            // iterate through the pixel values of each digit within the csv
        {
            // If it's not the first pixel in the row, expect a comma before the value.
            if (pixel > 0)
            {
                fscanf(file, ",");
            }
            fscanf(file, "%lf", &pixel_intensity);
            dataset[example].image[pixel] = pixel_intensity;
        }
    }

    fclose(file);
}

void initialize_dataset(example* dataset)
{
    for (int i = 0; i < 100; i++)   // initialize all images within the dataset 
    {
        for (int j = 0; j < 784; j++)
        {
            dataset[i].image = malloc(784 * sizeof(double));
            check_memory_allocation(dataset[i].image);
        }
    }
}

