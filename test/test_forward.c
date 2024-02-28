#include "lib.h"



/**
 * @test test_newOutputVector() ensures that the array of Value ptrs returned by newOutputVector() are
 * initialized to zero
*/
void test_newOutputVector(void){

    printf("test_newOutputVector()...");

    Value** output = newOutputVector(15);
    assert(output != NULL);

    // validate init
    for (int i=0; i<15; i++){
        assert(output[i]->value == 0);
    }
    
    for (int i=0; i<15; i++){
        freeValue(&output[i]);
    } 
    free(output);

    printf("PASS!\n");
}

/**
 * @test test_MultiplyWeights() tests that the MultiplyWeights() function from forward.c correctly 
 * computes matrix vector multiplication using autograd.c Value operations
*/
void test_MultiplyWeights(void){

    printf("test_MultiplyWeights()...");

    // init input vector
    int inputSize = 5;
    Value** input = newOutputVector(inputSize);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;  
    input[3]->value = 4;
    input[4]->value = 5;

    // create graph stack for the operations
    GraphStack* graphStack = newGraphStack();

    // create layer
    int outputSize = 3;
    Layer* layer = newLayer(inputSize, outputSize);

    // reset all weights to 1 for testing purposes
    for(int i = 0; i < 15; i++){
        layer->weights[i]->value = 1;
    }

    // multiply weights
    Value** output = MultiplyWeights(layer, input, graphStack);

    // check that dot product of input vector w/ones vector equals 15
    assert(output[0]->value == 15);
    assert(output[1]->value == 15);
    assert(output[2]->value == 15);


    //cleanup
    releaseGraph(graphStack);
    freeLayer(&layer);

    printf("PASS!\n");
}

/**
 * @test test_AddBias() checks that the elementwise addition of arrays of Value ptrs done by AddBias()
 * is carried out as expected.
*/
void test_AddBias(void){

    printf("test_AddBias()...");    

    // init input vector
    int inputSize = 5;
    Value** input = newOutputVector(inputSize);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;  
    input[3]->value = 4;
    input[4]->value = 5;

    // create graph stack for the operations
    GraphStack* graphStack = newGraphStack();

    // create layer
    int outputSize = 5;
    Layer* layer = newLayer(inputSize, outputSize);

    // reset all biases to 1 for testing purposes
    for(int i = 0; i < 5; i++){
        layer->biases[i]->value = 1;
    }

    // Add biases to input vector
    Value** output = AddBias(layer, input, graphStack);

    // validate output when vector of ones added to input 
    assert(output[0]->value == 2);
    assert(output[1]->value == 3);
    assert(output[2]->value == 4);
    assert(output[3]->value == 5);
    assert(output[4]->value == 6);

    // cleanup
    releaseGraph(graphStack);
    freeLayer(&layer);

    printf("PASS!\n");
}

/**
 * @test test_ApplyReLU() tests that relu is applied to an output vector as is expected when calling ApplyReLU()
*/
void test_ApplyReLU(void){

    printf("test_ApplyReLU()...");    

    // init input vector
    int inputSize = 5;
    Value** input = newOutputVector(inputSize);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;  
    input[3]->value = 4;
    input[4]->value = 5;

    // create graph stack for the operations
    GraphStack* graphStack = newGraphStack();

    // create layer
    int outputSize = 5;
    Layer* layer = newLayer(inputSize, outputSize);

    // Add biases to input vector
    Value** output = ApplyReLU(layer, input, graphStack);

    // validate output when vector of ones added to input 
    assert(output[0]->value == 1);
    assert(output[1]->value == 2);
    assert(output[2]->value == 3);
    assert(output[3]->value == 4);
    assert(output[4]->value == 5);

    // cleanup
    releaseGraph(graphStack);
    freeLayer(&layer);

    printf("PASS!\n");
}

/**
 * @test test_Forward() tests to make sure backpropgation will run following an mlp forward pass, As well as making 
 * sure that calling backward on a scalar outp from the forward pass doesn't break the program.
*/
void test_Forward(void){

    printf("test_Forward()...");

    // create a new mlp
    int inputSize = 3;
    int layerSizes[] = {16, 8, 4, 2};
    int numLayers = 4;
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // init input vector
    Value** input = newOutputVector(inputSize);
    input[0]->value = 1;
    input[1]->value = 2;
    input[2]->value = 3;  

    // run forward pass
    Value** output = Forward(mlp, input);
    assert(output != NULL);


    // sum ouput vector to test backprop works on layer output
    Value* sum = Add(output[0], output[1], mlp->graphStack);

    // backpropagate gradient
    Backward(sum);

    // cleanup
    releaseGraph(mlp->graphStack);
    freeMLP(&mlp);

    printf("PASS!\n");
}   


/**
 * @test test_repeatedBackward() tests to make sure that backward can be called multiple times following 
 * a call to releaseGraph(). This addressed issue #1
*/
void test_repeatedBackward(void){

    printf("test_repeatedBackward()...");

    // create a new mlp
    int inputSize = 3;
    int layerSizes[] = {16, 8, 4, 2};
    int numLayers = 4;
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // declare input and output arr
    Value** input, **output, *sum;

    for (int i = 0; i<6; i++){

        // create new input vector for next forward passs
        input = newOutputVector(inputSize);
        input[0]->value = 1;
        input[1]->value = 2;
        input[2]->value = 3;    

        output = Forward(mlp, input);

        // convert output to a single value
        sum = Add(output[0], output[1], mlp->graphStack);     

        // backpropagate gradient
        Backward(sum);  

        // release Graph
        releaseGraph(mlp->graphStack);
    }

    // cleanup
    freeMLP(&mlp);

    printf("PASS!\n");
}

int main(void){

    test_newOutputVector();
    test_MultiplyWeights();
    test_AddBias();
    test_ApplyReLU();
    test_Forward();
    test_repeatedBackward();

    return 0;
}