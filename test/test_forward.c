#include "macros.h"
#include "structs.h"
#include "libraries.h"


/**
 * @notice test_MultiplyWeights() tests gradient tracked matrix multiplication for a layer and it's input
*/
void test_MultiplyWeights(){

    // reusing initOutputVector() for input
    Value** input = initOutputVector(3);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;

    // create layer
    Layer* layer = malloc(sizeof(Layer));
    layer->inputSize = 3;
    layer->outputSize = 4;
    layer->weights = initWeights(3, 4);   
    layer->outputVector = initOutputVector(4);

    // I'm going to reset all weights to 1 for testing purposes
    for(int i = 0; i < 12; i++){
        layer->weights[i]->value = 1;
    }

    MultiplyWeights(layer, input);

    assert(layer->outputVector[0]->value == 6);
    assert(layer->outputVector[1]->value == 6);
    assert(layer->outputVector[2]->value == 6);

    // Small backprop test 
    // We'll sum together the output vector and backpropagate the sum to the input
    // @note: this is  not an extensive test for backprop. More so just to check that the function runs
    Value* sum = Add(layer->outputVector[2], Add(layer->outputVector[0], layer->outputVector[1]));
    Backward(sum);
}   

int main(void){

    test_MultiplyWeights();

    return 0;
}