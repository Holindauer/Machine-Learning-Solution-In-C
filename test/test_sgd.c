#include "structs.h"


/**
 * @test test_learningRule tests to make sure that the learning rule is applied to the weights and biases of the mlp
 * @dev the test will set all weights and biases (and their respective gradients) to 1 and then apply the learning 
 * rule with a learning rate of 1. This should zero out the weights and biases
*/
void test_learningRule(void){

    // init mlp
    MLP* mlp = createMLP(3, (int[]){3, 2, 1}, 3);

    // set all weights, biases and gradients to 1
    Layer* layer = mlp->inputLayer;

    for(int i =0; i<mlp->numLayers; i++){

        // weights
        for(int j=0; j<layer->inputSize * layer->outputSize; j++){
            layer->weights[j]->value = 1;
            layer->weights[j]->grad = 1;
        }
        // biases
        for(int j=0; j<layer->outputSize; j++){
            layer->biases[j]->value = 1;
            layer->biases[j]->grad = 1;
        }

        layer = layer->next;
    }

    // applying the learning rule w/ lr=1 should zero out the weights and biases
    Step(mlp, 1);

    // retrieve the input layer again
    layer = mlp->inputLayer;

    // check that the weights and biases are zero
    for (int i=0; i<mlp->numLayers; i++){
        // check weights
        for (int j=0; j<layer->inputSize * layer->outputSize; j++){
            assert(layer->weights[j]->value == 0);
        }
        // check biases
        for (int j=0; j<layer->outputSize; j++){
            assert(layer->biases[j]->value == 0);
        }
        layer = layer->next;
    }

    freeMLP(mlp);
}   



int main(void){


    printf("Running sgd.c tests\n");

    test_learningRule();

    printf("All sgd.c tests passed\n");

}