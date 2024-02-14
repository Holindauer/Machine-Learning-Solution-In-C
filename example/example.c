#include "structs.h"
#include "macros.h"

#define EPOCHS 15
#define LR 0.001

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

    int example = 1;

    Forward(mlp, features[example]);

    Backward(mlp->outputLayer->outputVector[0]);

    Step(mlp, LR);

    zeroGrad(&mlp, inputSize, layerSizes, numLayers);


    printf("\n%lf", mlp->outputLayer->outputVector[0]->value);


    Forward(mlp, features[example+1]);

    assert(mlp->outputLayer->outputVector[0] != NULL);
    printf("\n%lf", mlp->outputLayer->outputVector[0]->value);


    /**
     * @bug This should not raise a segfault, but it
    */
    // Backward(mlp->outputLayer->outputVector[0]);

    
    

    // free memory when done
    freeDataFeatures(features, IRIS_ROWS);
    freeDataTargets(targets, IRIS_ROWS);
    freeMLP(mlp);

 
    return 0;
}
