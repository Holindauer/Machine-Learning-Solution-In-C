#include "structs.h"
#include "macros.h"

#define EPOCHS 15

int main(void){

    // declare 2D feature and target arrays to hold the iris data
    // Data is represented as Value structs to allow for automatic differentiation
    Value* features[IRIS_ROWS][IRIS_FEATURES];
    Value* targets[IRIS_ROWS][IRIS_CLASSES];

    // load the iris data into the feature and target arrays
    loadData(features, targets);

    // create a multi-layer perceptron
    int inputSize = 4;
    int layerSizes[] = {16, 8, 4, 1};
    int numLayers = 4;

    // create the multi-layer perceptron
    MLP* mlp = createMLP(inputSize, layerSizes, numLayers); 

    // set up input vector
    Value** input = (Value**) malloc(IRIS_FEATURES * sizeof(Value*));
    for (int i = 0; i < IRIS_FEATURES; i++){
        input[i] = newValue(features[0][i]->value, NULL, NO_ANCESTORS, "input");
    }

    Forward(mlp, input);

    printf("Output: %f\n", mlp->outputLayer->outputVector[0]->value);

    Backward(mlp->outputLayer->outputVector[0]);


    
    // free memory when done
    freeDataFeatures(features, IRIS_ROWS);
    freeDataTargets(targets, IRIS_ROWS);
    freeMLP(mlp);

    // free the input vector
    for(int i = 0; i < inputSize; i++){
        freeValue(input[i]);
    }
    free(input);

    return 0;
}