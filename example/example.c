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

    // Set up an input vector
    Value** input;
    input = (Value**) malloc(IRIS_FEATURES * sizeof(Value*));
    for (int i = 0; i < IRIS_FEATURES; i++){
        input[i] = newValue(features[0][i]->value, NULL, NO_ANCESTORS, "input");
    }

    zeroGrad(mlp);

    // run forward pass
    Forward(mlp, input);

    // backpropagate gradient
    Backward(mlp->outputLayer->outputVector[0]);

    // for (int i =0; i<mlp->inputLayer->)



    // for (int epoch = 0; epoch < EPOCHS; epoch++){

    //     printf("Epoch: %d\n", epoch+1);

    //     input = (Value**) malloc(IRIS_FEATURES * sizeof(Value*));
    //     for (int i = 0; i < IRIS_FEATURES; i++){
    //         input[i] = newValue(features[0][i]->value, NULL, NO_ANCESTORS, "input");
    //     }

    //     // // run forward pass
    //     Forward(mlp, input);

    //     // backpropagate gradient
    //     Backward(mlp->outputLayer->outputVector[0]);

    //     // appply gradient descent
    //     Step(mlp, LR);

    //     // zero gradient
    //     zeroGrad(mlp);


    //     assert(mlp->outputLayer->outputVector[0] != NULL);


    //     // release computation graph once gradient has been accumulated
    //     releaseGraph(&mlp->outputLayer->outputVector[0]);

    //     assert(mlp->outputLayer->outputVector[0] != NULL);

    //     // mlp->outputLayer->outputVector[0] = newValue(0, NULL, NO_ANCESTORS, "initOutputVector"); // <--- this is a quick and dirty fix to fix releaseGraph() deallocating the outputVector of the mlp
    //     // // It fixes the prblem with not being able to run the forward  ass int he second epoch. But causes backward now to work

    //     // free the input vector
    //     for(int i = 0; i < inputSize; i++){
    //         freeValue(input[i]);
    //     }
    //     free(input);
    // }

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