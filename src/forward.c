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