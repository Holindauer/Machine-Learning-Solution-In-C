#pragma once
#include "value.h"
#include "graphStack.h"

// mlp.h

/**
 * @note the Layer struct represents a single layer in an mlp 
*/
typedef struct _layer {

    // layer dims
    int inputSize;
    int outputSize;
    
    // weight and biase matrices/vectors
    Value** weights;
    Value** biases;

    // links to next and prev layers
    struct _layer* next;
    struct _layer* prev;
}Layer;


/**
 * @note the MLP struct represents a multi layer perceptron neural network as a linked list of Layer structs 
*/
typedef struct {

    int numLayers;

    // links to head and tail of mlp list
    Layer* inputLayer;
    Layer* outputLayer;

    // stack of the computational graph build up from applying operations from autoGrad.c
    GraphStack* graphStack;
}MLP;


// mlp functions
Layer* newLayer(int inputSize, int outputSize);
void freeLayer(Layer** layer);
MLP* newMLP(int inputSize, int layerSizes[], int numLayers);
void freeMLP(MLP** mlp);