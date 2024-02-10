#include "libraries.h"
#include "macros.h"
#include "structs.h"

/**
 * @notice sgd.c contains the implementation of the stochastic gradient descent algorithm for training
 * mlps defined in mlp.c and forward.c
 * 
*/



/**
 * @notice Step() is a helper function for the Step() function. It computes the gradient descent learning 
 * rule for the Step function to update the weights and biases of the mlp.
 * @param mlp The multi-layer perceptron to update
 * @param lr The learning rate for the update
*/
void Step(MLP* mlp, int lr){

    assert(mlp != NULL);

    // retrieve the input layer
    Layer* layer = mlp->inputLayer;

    // iterate layers
    for (int i=0; i<mlp->numLayers; i++){

        for (int j=0; j<layer->inputSize * layer->outputSize; j++){

            //apply learning rule to the weight
            layer->weights[j]->value -= lr * layer->weights[j]->grad;
        }
        for (int j=0; j<layer->outputSize; j++){

            // apply learning rule to the bias
            layer->biases[j]->value -= lr * layer->biases[j]->grad;
        }

        // move to the next layer
        layer = layer->next;
    }
}


