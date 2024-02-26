#include "lib.h"


/**
 * @test test_newLayer() tests that Layer structs allocated for in newLayer() are properly initilized to [-1, 1]
*/
void test_newLayer(void){

    printf("test_newLayer...");

    int inputSize = 4;
    int outputSize = 20;

    // create new layer
    Layer* layer = newLayer(inputSize, outputSize);
    assert(layer != NULL);

    // validate struct fields
    assert(layer->next == NULL);
    assert(layer->prev == NULL);
    assert(layer->inputSize == inputSize);
    assert(layer->outputSize == outputSize);

    // validate weight and bias initialization range
    for (int i = 0; i < (inputSize * outputSize); i++){
        assert(layer->weights[i]->value > -1 || layer->weights[i]->value < 1);
    }

    for (int i = 0; i < outputSize; i++){
        assert(layer->biases[i]->value > -1 || layer->biases[i]->value < 1);
        assert(layer->output[i]->value == 0);
    }

    // cleanup
    freeLayer(&layer);
    assert(layer == NULL);

    printf("PASS!\n");
}



/**
 * @test test_newMLP() tests that the newMLP() function from mlp.c properly initlizes the mlp linked list of Layers
 *  
*/
void test_newMLP(void){

    printf("test_newMLP...");

    // create mlp
    int inputSize = 4;
    int layerSizes[] = {16, 8, 4, 1};
    int numLayers = 4;
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // validate member init
    assert(mlp->numLayers = numLayers);

    // input layer sizes
    assert(mlp->inputLayer->inputSize == inputSize);
    assert(mlp->inputLayer->outputSize = layerSizes[0]);

    // output layer sizes
    assert(mlp->outputLayer->inputSize == layerSizes[2]);
    assert(mlp->outputLayer->outputSize == layerSizes[3]);

    // validate graph stack init
    assert(mlp->graphStack->len == 1);
    assert(mlp->graphStack->head->next == NULL);
    assert(mlp->graphStack->head->pValStruct == NULL);

    // validate layer initializations across mlp
    Layer* layer = mlp->inputLayer;
    assert(layer != NULL);

    while (layer != NULL){

        // validate weights
        for (int i=0; i<(layer->outputSize * layer->inputSize); i++){
            assert(layer->weights[i]->value > -1 || layer->weights[i]->value < 1);
        }

        // validate biases and output vectors
        for (int i=0; i<layer->outputSize; i++){
            assert(layer->biases[i]->value > -1 || layer->biases[i]->value < 1);
            assert(layer->output[i]->value == 0);
        }

        layer = layer->next;
    }

    // cleanup
    freeMLP(&mlp);
    assert(mlp == NULL);

    printf("PASS!\n");
}


int main(void){

    test_newLayer();
    test_newMLP(); 

    return 0;
}