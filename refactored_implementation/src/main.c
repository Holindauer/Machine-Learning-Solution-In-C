#include "functions.h"
#include "structs.h"
#include "libraries.h"



int main(void) {

	// seed random
	srand((unsigned int)time(NULL));

	// load in iris dataset
	Data dataset;
	init_dataset(&dataset);

	FILE* stream = fopen("onehot_iris.csv", "r");
	load_data(&dataset, stream);
	print_dataset(dataset);


	// init model
	Model model;
	init_model(&model);

	// init gradient accumulator
	Gradient gradient_accumulator;
	zero_init_gradient_accumulator(&gradient_accumulator);

	// init hyperparameters
	double learning_rate = 0.001;
	int epochs = 100;
	
	// init training stats
	double loss = 0, accuracy = 0;
	
	// training loop
	for (int epoch = 0; epoch < epochs; epoch++) {

		for (int example = 0; example < NUM_EXAMPLES; example++) {

			// zero gradients
			zero_init_gradient_accumulator(&gradient_accumulator);

			// run forward pass
			model.prediction[example] = forward(&model, dataset.features[example], example);

			// backward pass
			backprop(&model, &dataset, example);

			// accumulate epoch gradient into the gradient_accumulator
			Accumulate_Gradient(&model, &gradient_accumulator);

		}

		// update weights
		Stochastic_Gradient_Descent(&model, &gradient_accumulator, learning_rate);

		// compute training metrics and loss for the epoch
		loss = Categorical_Cross_Entropy(model.output, dataset.targets);
		accuracy = Accuracy(model.prediction, dataset.targets);
		printf("\n Epoch: %d --- Loss: %lf --- Accuracy: %lf", epoch, loss, accuracy);

		Zero_Matrix(model.prediction, NUM_EXAMPLES, 1);


	}

	

	


	return 0;
}