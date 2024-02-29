#include "lib.h"

// gradientDescent.c

/**
 * @note Step() applies the gradient descent learning rule to an mlp 
 * @dev Step() is meant to be called directly after a call to Backward()
 * @dev weight and bias updates are performed in place on an mlp
 * @param mlp a ptr to an MLP struct to apply gradient descent to 
 * @param lr the learning rate to use in the update rule
*/
void Step(MLP* mlp, double lr){
    assert(mlp != NULL);

    // Retrieve input layer
    Layer* layer = mlp->inputLayer;

    while (layer != NULL){

        // update weights
        for (int i=0; i<(layer->inputSize * layer->outputSize); i++){
            
            layer->weights[i]->value -= lr * layer->weights[i]->grad;
        }

        // update biases
        for (int i=0; i<layer->outputSize; i++){
            
            layer->biases[i]->value -= lr * layer->biases[i]->grad;
        }

        layer = layer->next;
    }
}