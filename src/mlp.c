#include "libraries.h"
#include "macros.h"
#include "structs.h"


/**
 * @notice mlp.c contains the implementation of a constructor for a multi-layer perceptron
 * @dev Each layer of an mlp is a node in a doubly linked list
 * @dev Each layer contains a weight matrix and a bias vector. These parameters are represented as
 * Value structs in order to support backpropagation using the autograd implementation from autoGrad.c
*/


/**
 * @notice initWeights() is a helper function for the createMLP() constructor. It initializes the weights
 * (Value structs) for a given layer with random values between -1 and 1. 
*/
Value** initWeights(int inputSize, int outputSize){

    Value** weights = (Value**)malloc(inputSize * outputSize * sizeof(Value*));

    for(int i = 0; i < (inputSize * outputSize); i++){

        // create random value between -1 and 1
        float randomFloat = (float)rand() / (RAND_MAX + 1u) * 2.0f - 1.0f;
        weights[i] = newValue(randomFloat, NULL, NO_ANCESTORS, "initWeights");

    }

    return weights;
}

/**
 * @notice initBiases() is a helper function for the createMLP() constructor. It initializes the biases
 * (Value structs) for a given layer with random values between -1 and 1. 
*/
Value** initBiases(int outputSize){

    Value** biases = (Value**)malloc(outputSize * sizeof(Value*));

    for(int i = 0; i < outputSize; i++){

        // create random value between -1 and 1
        float randomFloat = (float)rand() / (RAND_MAX + 1u) * 2.0f - 1.0f;
        biases[i] = newValue(randomFloat, NULL, 0, "initBiases");
    }

    return biases;
}

/**
 * @notice createMLP() is a constructor function that creates a multi-layer perceptron with the specified
 * layer sizes. The weights and biases are initialized with random values between -1 and 1. 
 * @dev The MLP struct is used to store the Layer structs in a doubly linked list. The head and tail pointers
 * are used to access the start and end of the network.
 * @param inputSize The length of the input feature vector
 * @param layerSizes An array of integers representing the number of neurons in each layer of the network
*/
MLP* createMLP(int inputSize, int layerSizes[], int numLayers){

    // allocate memory for the MLP struct
    MLP* mlp = (MLP*)malloc(sizeof(MLP));  
    assert(mlp != NULL);

    // allocate memory for the input layer
    Layer* inputLayer = (Layer*)malloc(sizeof(Layer));
    assert(inputLayer != NULL);

    // set input layer fields
    inputLayer->inputSize = inputSize;  
    inputLayer->outputSize = layerSizes[0];
    inputLayer->prev = NULL;

    // initialize weights and biases for input layer
    inputLayer->weights = initWeights(inputSize, layerSizes[0]);
    inputLayer->biases = initBiases(layerSizes[0]);
    assert(inputLayer->weights != NULL);
    assert(inputLayer->biases != NULL);

    // set links for the input layer of mlp
    mlp->inputLayer = inputLayer;
    mlp->numLayers = numLayers;

    // allocate create the rest of the layers
    for (int i=1; i<numLayers; i++){

        // allocate memory for the current layer
        Layer* currentLayer = (Layer*)malloc(sizeof(Layer));
        assert(currentLayer != NULL);

        // set current layer fields
        currentLayer->inputSize = layerSizes[i-1];
        currentLayer->outputSize = layerSizes[i];
        currentLayer->prev = inputLayer;

        // initialize weights and biases for current layer
        currentLayer->weights = initWeights(layerSizes[i-1], layerSizes[i]);
        currentLayer->biases = initBiases(layerSizes[i]);
        assert(currentLayer->weights != NULL);
        assert(currentLayer->biases != NULL);

        // set next pointer for input layer
        inputLayer->next = currentLayer;

        // set links for the most recent layer
        currentLayer->next = NULL;
        currentLayer->prev = inputLayer;

        // update input layer pointer to current layer
        inputLayer = currentLayer;
    }

    // set output layer pointer
    mlp->outputLayer = inputLayer;

    return mlp;
}

/**
 * @notice freeWeights() is a helper function for the createMLP() destructor. It frees the memory allocated
 * for the weights of a given layer.
*/
void freeWeights(Value** weights, int inputSize, int outputSize){
    for(int i =0; i < (inputSize * outputSize); i++){
        freeValue(weights[i]);
    }
    free(weights);
}

/**
 * @notice freeBiases() is a helper function for the createMLP() destructor. It frees the memory allocated
 * for the biases of a given layer.
*/
void freeBiases(Value** biases, int outputSize){
    for(int i =0; i < outputSize; i++){
        freeValue(biases[i]);
    }
    free(biases);
}

/**
 * @notice freeMLP() is a destructor function that frees the memory allocated for the MLP struct and all of its
 * associated layers, weights, and biases.
*/
void freeMLP(MLP* mlp){

    Layer* currentLayer = mlp->inputLayer;
    Layer* nextLayer;

    while(currentLayer != NULL){
        nextLayer = currentLayer->next;
        freeWeights(currentLayer->weights, currentLayer->inputSize, currentLayer->outputSize);
        freeBiases(currentLayer->biases, currentLayer->outputSize);
        free(currentLayer);
        currentLayer = nextLayer;
    }

    free(mlp);
}
