#include "functions.h"
#include "structs.h"
#include "libraries.h"



void check_memory_allocation(double* arr)
{
	if (!arr) {
		fprintf(stderr, "Memory allocation failed!\n");
		exit(EXIT_FAILURE);  // Exit the program.
	}
}

void free_dataset(example* dataset, int num_examples)
{
	for (int i = 0; i < num_examples; i++)
	{
		free(dataset[i].image);
	}

}