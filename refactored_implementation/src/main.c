#include  "functions.h"
#include "structs.h"
#include "libraries.h"



int main(void) {

	// load in iris dataset
	Data dataset;
	init_dataset(&dataset);

	FILE* stream = fopen("onehot_iris.csv", "r");
	load_data(&dataset, stream);
	print_dataset(dataset);


	// init model
	Model model;
	init_model(&model);




	return 0;
}