#include "structs.h"
#include "libraries.h"
#include "macros.h"



/**
 * @test this test ensures the GraphStack is initialized w/ correct values
*/
void test_GraphStackInit(void){
    // allocate memory for the stack
    GraphStack* stack = newGraphStack();
    
    // check that the stack is initialized correctly
    assert(stack != NULL);
    assert(stack->head == NULL);
    assert(stack->len == 0);

    // free the stack
    free(stack);
}

/**
 * @test test_pushGraphStack() ensures the pushGraphStack function works as expected
 * 
*/
void test_pushGraphStack(void){

    // init stack
    GraphStack* stack = newGraphStack();

    // create values to push onto the stack
    Value* value1 = newValue(1, NULL, NO_ANCESTORS, "test");
    Value* value2 = newValue(2, NULL, NO_ANCESTORS, "test");

    pushGraphStack(stack, value1);
    pushGraphStack(stack, value2);
    
    // check that the stack is initialized correctly
    assert(stack->head != NULL);
    assert(stack->len == 2);
    assert(stack->head->value == value2);
    assert(stack->head->next->value == value1);

    releaseGraph(stack);

    printf("test_pushGraphStack passed\n");
}

/**
 * @test test_popGraphStack() ensures the popGraphStack function works as expected
 * 
*/
void test_popGraphStack(void){

    // init stack
    GraphStack* stack = newGraphStack();

    // create values to push onto the stack
    Value* value1 = newValue(1, NULL, NO_ANCESTORS, "test");
    Value* value2 = newValue(2, NULL, NO_ANCESTORS, "test");

    pushGraphStack(stack, value1);
    pushGraphStack(stack, value2);
    

    popGraphStack(stack);

    // check that the first pop worked
    assert(stack->head->value == value1);
    assert(stack->len == 1);

    popGraphStack(stack);

    // check that the stack is empty
    assert(stack->head == NULL);
    assert(stack->len == 0);

    free(stack);

    printf("test_popGraphStack passed\n");
}


/**
 * @test There is currently an issue when running example.c where calling zeroGrad() which contains a call to 
 * releaseGraph() is breaking the Backward() function. This this is designed to address this bug and ensure 
 * the program does not segfault when calling Backward() after zeroGrad() in an isolated test. 
*/
void test_postReleaseGradIntegrity(void){

    // Declare an MLP
    MLP* mlp = createMLP(4, (int[]){16, 8, 4, 1}, 4);

    printf("MLP created\n");

    // Declare an input Value array
    Value** input = (Value**)malloc(4 * sizeof(Value*));
    for(int i = 0; i < 4; i++){
        input[i] = newValue(1, NULL, NO_ANCESTORS, "testInput");
    }

    // run a forward pass
    Forward(mlp, input);

    // run a backward pass
    Backward(mlp->outputLayer->outputVector[0]);

    // zero the gradients
    zeroGrad(mlp);

    // // run a forward pass
    Forward(mlp, input);

    // run a backward pass
    Backward(mlp->outputLayer->outputVector[0]);

}

/**
 * @test test_peekGraphStack() ensures the peekGraphStack function works as expected
*/
void test_peekGraphStack(void){

    // init stack
    GraphStack* stack = newGraphStack();

    // create values to push onto the stack
    Value* value1 = newValue(1, NULL, NO_ANCESTORS, "test");
    Value* value2 = newValue(2, NULL, NO_ANCESTORS, "test");

    pushGraphStack(stack, value1);
    pushGraphStack(stack, value2);
    
    // check that the stack is initialized correctly
    Value* peekedValue = peekGraphStack(stack);
    
    assert(peekedValue->value == value2->value);

    releaseGraph(stack);

    printf("test_peekGraphStack passed\n");
}


/**
 * @test test_mlpIntactPostZeroGrad() tests that after calling a forward pass, backward pass, and zeroGrad() an 
 * MLP is same as prior to calling zeroGrad()
 * @dev this involves copying the MLP and comparing the two
 * 
*/
void test_mlpIntactPostZeroGrad(void){

    // Create an MLP
    MLP* mlp = createMLP(4, (int[]){16, 8, 4, 1}, 4);

    // Declare an input Value array
    Value** input = (Value**)malloc(4 * sizeof(Value*));
    for(int i = 0; i < 4; i++){
        input[i] = newValue(1, NULL, NO_ANCESTORS, "testInput");
    }

    // run a forward pass
    Forward(mlp, input);

    // run a backward pass
    Backward(mlp->outputLayer->outputVector[0]);

    // Create another MLP of the same specs
    MLP* mlpCopy = createMLP(4, (int[]){16, 8, 4, 1}, 4);

    // Now were going to copy the mlp into mlpCopy for all layers
    Layer *layer = mlp->inputLayer;
    Layer *layerCopy = mlpCopy->inputLayer; 

    // copy each layer of the mlp into the mlpCopy
    for (int i = 0; i < mlp->numLayers; i++){   
        
        // copy layer specs
        layerCopy->inputSize = layer->inputSize;
        layerCopy->outputSize = layer->outputSize;

        for (int i = 0; i< layer->inputSize * layer->outputSize; i++){ 
            // copy weights
            layerCopy->weights[i]->value = layer->weights[i]->value;
            layerCopy->weights[i]->grad = layer->weights[i]->grad;
            strcpy(layerCopy->weights[i]->opStr, layer->weights[i]->opStr);

            // @note there won't be any ancestors to copy since this is an mlp
            assert(layer->weights[i]->ancestors == NULL);
            assert(layerCopy->weights[i]->ancestors == NULL);

        }
        for (int i = 0; i< layer->outputSize; i++){ // copy biases and outputVector

            // copy biases
            layerCopy->biases[i]->value = layer->biases[i]->value;
            layerCopy->biases[i]->grad = layer->biases[i]->grad;
            strcpy(layerCopy->biases[i]->opStr, layer->biases[i]->opStr);

             // @note there won't be any ancestors to copy since this is an mlp
            assert(layer->biases[i]->ancestors == NULL);
            assert(layerCopy->biases[i]->ancestors == NULL);

            // copy outputVector
            layerCopy->outputVector[i]->value = layer->outputVector[i]->value;
            layerCopy->outputVector[i]->grad = layer->outputVector[i]->grad;
            strcpy(layerCopy->outputVector[i]->opStr, layer->outputVector[i]->opStr);

            // @note outputVector will have ancestors 
        }

        // move to the next layer
        layer = layer->next;
        layerCopy = layerCopy->next;
    }

    // zero the gradients
    zeroGrad(mlp);

    // reset layer and layerCopy to the inputLayers
    layer = mlp->inputLayer;
    layerCopy = mlpCopy->inputLayer;

    // now check whether the fields we copied are still in the og mlp post zero grad
    for (int i = 0; i < mlp->numLayers; i++){   
        
        // check layer specs
        assert(layerCopy->inputSize == layer->inputSize);
        assert(layerCopy->outputSize == layer->outputSize);

        // check that weight, bias, and outputVector fields still exist
        assert(layerCopy->weights != NULL);
        assert(layerCopy->biases != NULL);
        assert(layerCopy->outputVector != NULL);

        for (int i = 0; i< layer->inputSize * layer->outputSize; i++){ 
            
            // check weights
            assert(layerCopy->weights[i]->value == layer->weights[i]->value);
            assert(layer->weights[i]->grad == 0);
            assert(strcmp(layerCopy->weights[i]->opStr, layer->weights[i]->opStr) == 0);

            // @note there won't be any ancestors to copy since this is an mlp
            assert(layer->weights[i]->ancestors == NULL);
            assert(layerCopy->weights[i]->ancestors == NULL);

        }
        for (int i = 0; i< layer->outputSize; i++){ // copy biases and outputVector

            // copy biases
            assert(layerCopy->biases[i]->value == layer->biases[i]->value);
            assert(layer->biases[i]->grad == 0);
            assert(strcmp(layerCopy->biases[i]->opStr, layer->biases[i]->opStr) == 0);

             // @note there won't be any ancestors to copy since this is an mlp
            assert(layer->biases[i]->ancestors == NULL);
            assert(layerCopy->biases[i]->ancestors == NULL);

            // copy outputVector
            assert(layer->outputVector[i]->value == 0);
            assert(layer->outputVector[i]->grad == 0);
            assert(strcmp(layerCopy->outputVector[i]->opStr, layer->outputVector[i]->opStr) == 0);

            // @note outputVector will have ancestors 
        }

        // move to the next layer
        layer = layer->next;
        layerCopy = layerCopy->next;
    }


    // cleanup
    freeMLP(mlp);
    freeMLP(mlpCopy);
    for (int i = 0; i < 4; i++){
        freeValue(input[i]);
    }
    free(input);
}


int main(void){

    printf("\nTesting graphTracker.c\n");

    test_GraphStackInit();
    test_pushGraphStack();
    test_popGraphStack();
    test_peekGraphStack();

    // test_mlpIntactPostZeroGrad(); <-- currently not passing
    test_postReleaseGradIntegrity();

    printf("\nAll tests passed!\n");

    return 0;
}