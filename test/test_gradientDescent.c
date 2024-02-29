#include "lib.h"


void test_Step(void){

    printf("test_Step()...");

    // create a new mlp
    int inputSize = 3;
    int layerSizes[] = {16, 8, 4, 2};
    int numLayers = 4;
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);


    // set all weights and biases, and gradients to 1
    Layer* layer = mlp->inputLayer;
    while (layer != NULL){

        for (int i=0; i< (layer->inputSize * layer->outputSize); i++){
            layer->weights[i]->value = 1;
            layer->weights[i]->grad = 1;
        }
        for (int i=0; i<layer->outputSize; i++){
            layer->biases[i]->value = 1;
            layer->biases[i]->grad = 1;
        }

        layer = layer->next;
    }

    // apply gradient descent with a learning rate of 1
    double lr = 1;
    Step(mlp, lr);


    // check that all weights are now zero
    layer = mlp->inputLayer;
    while (layer != NULL){

        for (int i=0; i< (layer->inputSize * layer->outputSize); i++){
            assert(layer->weights[i]->grad == 0);
        }
        for (int i=0; i<layer->outputSize; i++){
            assert(layer->weights[i]->grad == 0);
        }

        layer = layer->next;
    }

    // cleanup
    freeMLP(&mlp);

    printf("PASS!\n");
}




int main(void){

    test_Step();

    return 0;
}