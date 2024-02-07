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
//  * @notice Forward() is used to perform the forward pass of an mlp. 
//  * 
// */
// void Forward(MLP* mlp, Value** input){

//     // copy input vector so that when we deallocate the forward pass graph, we don't deallocate the training data
//     Value** inputCopy = (Value**)malloc(mlp->inputLayer->inputSize * sizeof(Value*));
//     for(int i = 0; i < mlp->inputLayer->inputSize; i++){
//         inputCopy[i] = newValue(input[i]->value, NULL, NO_ANCESTORS, "Forward");
//     }



// }