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

    printf("PASS!\n");
}


// /**
//  * @test test_Forward() tests to make sure backpropgation will run following an mlp forward pass 
// */
// void test_Forward(void){

//     printf("test_Forward()...");

//     // create a new mlp
//     int inputSize = 4;
//     int layerSizes[] = {16, 8, 4, 1};
//     int numLayers = 4;
//     MLP* mlp = newMLP(inputSize, layerSizes, numLayers);


//     // create a dummy input
//     Value** input = (Value**)malloc(sizeof(Value*) * inputSize);
//     assert(input != NULL);
//     for (int i=0; i<inputSize; i++){
//         input[i] = newValue(1, NULL, NO_ANCESTORS, "init input");
//         assert(input[i] != NULL);
//     }

//     printf("\nOutside of Forward Call");
//     Value** output = Forward(mlp, input);





//     freeMLP(&mlp);

//     for (int i=0; i<inputSize; i++){
//         freeValue(input[i]);
//     }
//     free(input);

//     printf("PASS!\n");
// }   


int main(void){

    test_newOutputVector();
    test_MultiplyWeights();
    test_AddBias();
    // test_Forward();

    return 0;
}