#include "functions.h"
#include "libraries.h"
#include "structs.h"

// used to iitialize the dataset struct 
void init_dataset(Data* dataset) {
	for (int example = 0; example < NUM_EXAMPLES; example++) {

		for (int feature = 0; feature < NUM_FEATURES; feature++) { // init features
			dataset->features[example][feature] = 69;
		}
		for (int class = 0; class < NUM_CLASSES; class++) { // init targets
			dataset->targets[example][class] = 69;
		}
	}
}

// prints dataset 
void print_dataset(Data dataset) {
	for (int example = 0; example < NUM_EXAMPLES; example++) {

		printf("\nExample: %d Features: ", example);
		for (int feature = 0; feature < NUM_FEATURES; feature++) { // init features
			printf("%.2f ", dataset.features[example][feature]);
		}
		printf("Target: ");
		for (int class = 0; class < NUM_CLASSES; class++) { // init targets
			printf("%.2f ", dataset.targets[example][class]);
		}
	}

}

// loads in iris data from csv
void load_data(Data* dataset, FILE* stream) {

	for (int example = 0; example < NUM_EXAMPLES; example++) {
		for (int feature = 0; feature < NUM_FEATURES; feature++) { 
			fscanf(stream, "%lf,", &dataset->features[example][feature]); // read features 
		}
		for (int class = 0; class < NUM_CLASSES; class++) { 
			fscanf(stream, "%lf,", &dataset->targets[example][class]); // read targets
		}
	}
}