#include "structs.h"
#include "macros.h"



int main(void){

    // declare 2D feature and target arrays to hold the iris data
    // Data is represented as Value structs to allow for automatic differentiation
    Value* features[IRIS_ROWS][IRIS_FEATURES];
    Value* targets[IRIS_ROWS][IRIS_CLASSES];

    // load the iris data into the feature and target arrays
    loadData(features, targets);

    // create a multi-layer perceptron
    int inputSize = 4;
    int layerSizes[] = {4, 16, 8, 1};
    int numLayers = 4;

    MLP* mlp = createMLP(inputSize, layerSizes, numLayers);

    // free memory when done
    freeDataFeatures(features, IRIS_ROWS);
    freeDataTargets(targets, IRIS_ROWS);
    freeMLP(mlp);

    return 0;
}