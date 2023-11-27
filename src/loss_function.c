#include "functions.h"
#include "structs.h"
#include "libraries.h"


double Categorical_Cross_Entropy(double output[][LAYER_2_NEURONS], double targets[][NUM_CLASSES]) {

	double loss_sum = 0, loss = 0;

	for (int example = 0; example < NUM_EXAMPLES; example++) { // sum loss across all predictions
		
		
		loss_sum = 0;
		for (int c = 0; c < NUM_CLASSES; c++) { // L = -C_sigma_c=1 [y_c * log(p_c)]
			loss_sum += -(targets[example][c] * log(output[example][c]));
		}

		loss += loss_sum;
	}

	return loss_sum;
}