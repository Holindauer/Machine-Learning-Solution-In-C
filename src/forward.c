#include "libraries.h"
#include "macros.h"
#include "structs.h"

/**
 * @notice forward.c implements the forward pass of the neural network
 */



/**
 * @notice MultiplyWeights() implements gradient tracked matrix multiplication for a layer and it's input
 * @dev MultiplyWeights() works for a single input vector, not a batch of input vectors.

*/
void MultiplyWeights(Layer* layer, Value** input){

    // iterate over each output neuron
    for (int i = 0; i < layer->outputSize; i++){

        // Accumulate the dot product of the i'th row of weights and the input vector
        for (int j = 0; j < layer->inputSize; j++){
            // Add to the i'th output the product of the weight and the input
            // Note: layer->weights is a 2D array reoresented as a 1D array
            layer->outputVector[i] = Add(
                layer->outputVector[i], 
                Mul(layer->weights[i * layer->inputSize + j], input[j]) 
                );
        }
    }
}


/**
 * @notice AddBias() adds the bias to the output vector of a layer after MultiplyWeights() has been called
 * @dev AddBias() works for a single input vector, not a batch of input vectors.
 * @dev AddBias() uses the autoGrad.c infrastructure to track the forward pass for backpropagation
*/
void AddBias(Layer* layer, Value** input){

    // iterate over each output neuron
    for (int i = 0; i < layer->outputSize; i++){

        // elementwise Add() the bias to the output vector
        layer->outputVector[i] = Add(layer->outputVector[i], input[i]);
    }
}


/**
 * @notice copyInput() is a helper function used to copy the input vector at the start of the forward pass so that
 * when we deallocate the forward pass graph, we do not inadvertently deallocate the training data at the base of it. 
*/
Value** copyInput(Value** input, int inputSize){
    Value** inputCopy = (Value**)malloc(inputSize * sizeof(Value*));
    for(int i = 0; i < inputSize; i++){
        inputCopy[i] = newValue(input[i]->value, NULL, NO_ANCESTORS, "copyInput");
    }
    return inputCopy;
}

/**
 * @notice Forward() is used to perform the forward pass of an mlp. 
 * @dev The final output of the network is stored in the outputVector memebr of the 
 * outputLayer member of the mlp struct
*/
void Forward(MLP* mlp, Value** input){

    // copy input vector to avoid deallocation of training data when calling releaseGraph()
    Value** inputCopy = copyInput(input, mlp->inputLayer->inputSize);
    assert(inputCopy != NULL);

    // retrieve the input layer
    Layer* layer = mlp->inputLayer;

    // pass input to the inputlayer
    MultiplyWeights(layer, inputCopy);
    AddBias(layer, layer->outputVector);

    // for subsequent layers iterate over the rest of the layers, 
    // passing the output of the previous layer to the next layer
    for (int i = 1; i < mlp->numLayers; i++){
        layer = layer->next;
        MultiplyWeights(layer, layer->prev->outputVector);
        AddBias(layer, layer->prev->outputVector);
    }
}