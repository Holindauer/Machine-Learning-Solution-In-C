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


    // zero the gradients of the weights and biases
    zeroGrad(mlp);

    // Forward pass
    Forward(mlp, input);

    // Backward pass
    Backward(mlp->outputLayer->outputVector[0]);

    // Update weights and biases
    Step(mlp, LR);

    // Release the Graph
    releaseGraph(&mlp->outputLayer->outputVector[0]);

    // check we havent deallocated the mlp
    assert(mlp->inputLayer->weights[0] != NULL);
    assert(mlp->outputLayer->outputVector[0] != NULL);

    // ensure we have correctly not deallocated the input vector
    assert(input[0] != NULL);



    // Set up an input vector
    Value** input2;
    input2 = (Value**) malloc(IRIS_FEATURES * sizeof(Value*));
    for (int i = 0; i < IRIS_FEATURES; i++){
        input2[i] = newValue(features[0][i]->value, NULL, NO_ANCESTORS, "input");
    }

        // zero the gradients of the weights and biases
    zeroGrad(mlp);

    // Forward pass
    Forward(mlp, input2);
    
    // Backward pass
    Backward(mlp->outputLayer->outputVector[0]);  

    // Update weights and biases
    Step(mlp, LR);

    // Release the Graph
    releaseGraph(&mlp->outputLayer->outputVector[0]);


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