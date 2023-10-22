#include "libraries.h"
#include "functions.h"
#include "structs.h"

/* This function loads in flattened mnist digits from a csv */
void load_data(const char* filename, example* dataset, int num_digits){
    //--------------------------------------------------------------------------------------------------------------------Open File
    FILE* file = fopen(filename, "r");                             
    if (file == NULL){
        fprintf(stderr, "Could not open file %s\n", filename);       
        exit(1);}

    //--------------------------------------------------------------------------------------------------------------------Read in Data

    int label = 0;             
    double pixel_intensity = 0; 

    for (int example = 0; example < num_digits; example++)   // loop through the specified number of digits to extract from the dataset  
    {
        fscanf(file, "%d,", &label);
        dataset[example].label = label;

        for (int pixel = 0; pixel < 784; pixel++)
        {
            // If it's not the first pixel in the row, expect a comma before the value.
            if (pixel > 0) { fscanf(file, ","); }

            fscanf(file, "%lf", &pixel_intensity);
            dataset[example].image[pixel] = pixel_intensity;
        }
    }
    fclose(file);
}

/* inititalizes each array in example in array of example stracts */
void initialize_dataset(example* dataset, int num_examples)
{
    for (int i = 0; i < num_examples; i++)  
    {
        for (int j = 0; j < 784; j++)
        {
            dataset[i].image = malloc(784 * sizeof(double));
            check_memory_allocation(dataset[i].image);
        }
    }
}

