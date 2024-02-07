#include "structs.h"

/**
 * @test test_initWeights() tests the initWeights() function from mlp.c   
 * @dev The function initializes the weights (Value structs) for a given layer with random values between -1 and 1.
*/
void test_initWeights(){

    int inputSize = 3;
    int outputSize = 2;

    Value** weights = initWeights(inputSize, outputSize);

    for(int i = 0; i < (inputSize * outputSize); i++){
            assert(weights[i]->value >= -1 && weights[i]->value <= 1);
            assert(weights[i]->grad == 0);
        }   
    
    freeWeights(weights, inputSize, outputSize);
}

/**
 * @test test_initBiases() tests the initBiases() function from mlp.c
 * @dev The function initializes the biases (Value structs) for a given layer with random values between -1 and 1.
*/
void test_initBiases(){

    int outputSize = 6;

    Value** biases = initBiases(outputSize);

    for(int i = 0; i < outputSize; i++){
        assert(biases[i]->value >= -1 && biases[i]->value <= 1);
        assert(biases[i]->grad == 0);
    }

    freeBiases(biases, outputSize);
}

/**
 * @test test_initOutputVector() tests the initOutputVector() function from mlp.c
 * @dev The function initializes the output vector for a given layer with Value structs that have a value of 0 and no ancestors.
*/
void test_initOutputVector(){

    int outputSize = 5;

    Value** output = initOutputVector(outputSize);

    for(int i = 0; i < outputSize; i++){
        assert(output[i]->value == 0);
        assert(output[i]->grad == 0);
    }

    freeBiases(output, outputSize);
}


/**
 * @notice test_mlpInit() tests the createMLP() function from mlp.c
*/
void test_mlpInit(){


    int inputSize = 4;
    int layerSizes[] = {4, 3, 2};
    int numLayers = 3;

    MLP* mlp = createMLP(inputSize, layerSizes, numLayers);

    // check that the mlp stats were initialized correctly 
    assert(mlp->inputLayer->inputSize == 4);
    assert(mlp->inputLayer->outputSize == 4);
    assert(mlp->inputLayer->next->outputSize == 3);
    assert(mlp->inputLayer->next->next->outputSize == 2);
    assert(mlp->outputLayer->outputSize == 2);
    assert(mlp->numLayers == 3);
    assert(mlp->inputLayer->next->next->next == NULL);

    // Ensure all weights and biases are properly initialized between -1 and 1
    Layer* currentLayer = mlp->inputLayer;
    while (currentLayer != NULL){
        // check that the weights and biases were initialized correctly
        for(int i = 0; i < (currentLayer->inputSize * currentLayer->outputSize); i++){
            assert(currentLayer->weights[i]->value >= -1 && currentLayer->weights[i]->value <= 1);
            assert(currentLayer->weights[i]->grad == 0);
        }
        // check that the biases and output vectors were initialized correctly
        for(int i = 0; i < currentLayer->outputSize; i++){
            assert(currentLayer->biases[i]->value >= -1 && currentLayer->biases[i]->value <= 1);
            assert(currentLayer->biases[i]->grad == 0);

            assert(currentLayer->outputVector[i]->value == 0);
            assert(currentLayer->outputVector[i]->grad == 0);
        }
        currentLayer = currentLayer->next;
    }

    freeMLP(mlp);
}




int main(void){

    printf("Testing mlp funcs...\n");

    test_initWeights();
    test_initBiases();
    test_mlpInit();
    
    printf("All tests passed!\n\n");

    return 0;
}