#include "libraries.h"
#include "structs.h"
#include "functions.h"

double Accuracy(double prediction[LAYER_2_NEURONS], double targets[][NUM_CLASSES]) {

	double True_Positives = 0;
	int predicted_argmax = 0;
	for (int example = 0; example < NUM_EXAMPLES; example++) {

		predicted_argmax = prediction[example];
		
		if (targets[example][predicted_argmax] == 1) {
			True_Positives += 1;
		}
	}

	return True_Positives / NUM_EXAMPLES;
}