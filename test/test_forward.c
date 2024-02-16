#include "macros.h"
#include "structs.h"
#include "libraries.h"


/**
 * @notice test_MultiplyWeights() tests gradient tracked matrix multiplication for a layer and it's input
*/
void test_MultiplyWeights(){

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();


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

    MultiplyWeights(layer, input, graphStack);

    assert(layer->outputVector[0]->value == 6);
    assert(layer->outputVector[1]->value == 6);
    assert(layer->outputVector[2]->value == 6);

    // Small backprop test 
    // We'll sum together the output vector and backpropagate the sum to the input
    // @note: this is  not an extensive test for backprop. More so just to check that the function runs
    Value* sum = Add(layer->outputVector[2], Add(layer->outputVector[0], layer->outputVector[1], graphStack), graphStack);
    Backward(sum);

    // free memory
    releaseGraph(graphStack);

    printf("MultiplyWeights() passed\n");
}   


/**
 * @notice test_AddBias() tests adding the bias to the output vector of a layer after MultiplyWeights() has been called
 * @dev AddBias() works for a single input vector, not a batch of input vectors.
 * @dev AddBias() uses the autoGrad.c infrastructure to track the forward pass for backpropagation
*/
void test_AddBias(void){

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();

    // init a vector of 3 values
    Value** input = initOutputVector(3);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;

    // create layer
    Layer* layer = malloc(sizeof(Layer));
    layer->inputSize = 3;
    layer->outputSize = 3;
    layer->outputVector = initOutputVector(3);

    // I'm going to reset all weights to 1 for testing purposes
    for(int i = 0; i < 3; i++){
        layer->outputVector[i]->value = 1;
    }

    // Add bias
    AddBias(layer, input, graphStack);

    assert(layer->outputVector[0]->value == 2);
    assert(layer->outputVector[1]->value == 3);
    assert(layer->outputVector[2]->value == 4);


    // Small backprop test
    // We'll sum together the output vector and backpropagate the sum to the input
    // @note: this is  not an extensive test for backprop. More so just to check that the function runs
    Value* sum = Add(layer->outputVector[2], Add(layer->outputVector[0], layer->outputVector[1], graphStack), graphStack);
    Backward(sum);

    // free memory
    releaseGraph(graphStack);

    printf("AddBias() passed\n");
}


/**
 * @notice test_Forward() tests that the forward pass of the network does not crash
*/
void test_Forward(){

    // create a new mlp
    int inputSize = 4;
    int layerSizes[] = {4, 3, 1};
    int numLayers = 3;
    MLP* mlp = createMLP(inputSize, layerSizes, numLayers);

    printf("MLP created\n");

    // create a new input vector
    Value** input = (Value**)malloc(inputSize * sizeof(Value*));
    for(int i = 0; i < inputSize; i++){
        input[i] = newValue(i, NULL, NO_ANCESTORS, "test_Forward");
    }

    printf("Input created\n");

    // run the forward pass
    Forward(mlp, input);   

    printf("Forward() ran\n");
    printf("Output: %f\n", mlp->outputLayer->outputVector[0]->value);
    printf("Output: %f\n", mlp->outputLayer->outputVector[0]->grad);
    

    // run a backward pass
    Backward(mlp->outputLayer->outputVector[0]);

    printf("Backward() ran\n");

    releaseGraph(mlp->graphStack);
    freeMLP(mlp);

    printf("Forward() passed\n");
}


int main(void){

    printf("\nRunning tests for forward.c\n");

    test_MultiplyWeights();
    test_AddBias();
    test_Forward();

    printf("All forward tests passed\n");

    return 0;
}