#include "autoGrad.h"
#include "lib.h"

/**
 * @helper checks if a new value with no ancestors was initialized properly
*/
void check_newValueInitialized(Value* value, int intendedValue, char opString[]){

    assert(value->ancestorArrLen == 0);
    assert(value->ancestors == NULL);
    assert(value->grad == 0);
    assert(value->value == intendedValue);
    assert(value->Backward == NULL);
    assert(strcmp(value->opString, opString) == 0);
}

/**
 * @test ensures new values are initialized properly and ancestor mechanism is working correctly
*/
void test_newValue(void){

    printf("test_newValue()...");

    // create three values and check they are initialized properly
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    check_newValueInitialized(a, 3, "a");

    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");
    check_newValueInitialized(a, 3, "a");

    Value* c = newValue(5, NULL, NO_ANCESTORS, "c");
    check_newValueInitialized(a, 3, "a");

    // manually set the ancestors of a new value to  a, b, and c
    Value* ancestors[3] = {a, b, c};
    int ancestorArrLen = 3;
    Value* v = newValue(9, ancestors, ancestorArrLen, "add");

    // check init of new value
    assert(v->value == 9);
    assert(v->grad == 0);
    assert(strcmp(v->opString, "add") == 0);

    // check ancestors are a, b, and c
    assert(v->ancestors[0] == a);
    assert(v->ancestors[1] == b);
    assert(v->ancestors[2] == c);

    // cleanup
    freeValue(&v);
    assert(v == NULL);

    freeValue(&a);
    assert(a == NULL);

    freeValue(&b);
    assert(b == NULL);

    freeValue(&c);
    assert(c == NULL);

    printf("PASS!\n");
}

/**
 * @helper check_emptyGraphStack() ensures that there is only one node in the graph stack and
 * that its fields are 0. Meant to check that releaseGraph() worked
*/
void check_emptyGraphStack(GraphStack* graphStack){
    assert(graphStack->len == 1);
    assert(graphStack->head != NULL);
    assert(graphStack->head->pValStruct == NULL);
    assert(graphStack->head->next == NULL);
}

/**
 * @helper check_graphStackAncestorUpdate() checks that the output Value from an autograd operation contains the 
 * expected fields.
 * @param opResult The Value struct that was the result of the operation
 * @param ancestor1 The first ancestor of the operation
 * @param ancestor2 The second ancestor of the operation (optiional if op isnt binary)
 * @param expectedValue The expected value of the operation
 * @param expectedGrad The expected gradient of the operation
 * @param ddx The expected type pBackwardFunc ptr to the backward function for the operation
 * @param opString The expected operation string (e.g. "add", "mull", "relu") 
 * @param binary A flag indicating if the operation is binary or not
*/
void check_opResultFields(
    Value* opResult, 
    Value* ancestor1, 
    Value* ancestor2, 
    double expectedValue, 
    double expectedGrad, 
    pBackwardFunc ddx, 
    char opString[], 
    int binary
    ){

    // assert value correctly added and related fields correctly updated
    assert(opResult->value == expectedValue);
    assert(opResult->grad == expectedGrad);
    assert(opResult->Backward == ddx);
    assert(strcmp(opResult->opString, opString) == 0);
    
    // check ancestors
    assert(opResult->ancestors[0] == ancestor1 || opResult->ancestors[1] == ancestor1);
    if (binary){
        assert(opResult->ancestors[0] == ancestor2 || opResult->ancestors[1] == ancestor2);
    }
}

/**
 * @helper check_graphStackUpdate() checks that the graph stack was updated correctly after an operation
 * by checking that the head of the graph stack is the result of the operation and that the length of the
 * graph stack matches expectations.
*/
void check_graphStackUpdate(GraphStack* graphStack, Value* expectedHead, int expectedLen){

    // assert resulting value correctly pushed to the graph stack
    assert(graphStack->head->pValStruct->value == expectedHead->value);
    assert(graphStack->head->pValStruct == expectedHead);
    assert(graphStack->len == expectedLen);
}


/**
 * @test test_Add() tests that the add operation is working correctly
*/
void test_Add(void){

    printf("test_Add()...");

    // init values to add
    Value* a = newValue(10, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(10, NULL, NO_ANCESTORS, "b");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // Add a and b
    Value* c = Add(a, b, graphStack);

    check_opResultFields(c, a, b, 20, 0, addBackward, "add", BINARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);
    
    printf("PASS!\n");
}

/**
 * @test test_Mul() tests that the Mul() operation is working correctly
*/
void test_Mul(void){

    printf("test_Mul()...");


    // init values to add
    Value* a = newValue(10, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(10, NULL, NO_ANCESTORS, "b");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // Mul a and b
    Value* c = Mul(a, b, graphStack);

    check_opResultFields(c, a, b, 100, 0, mulBackward, "mul", BINARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_ReLU() tests that the ReLU() operation is working correctly
*/
void test_ReLU(void){

    printf("test_ReLU()...");

    // init values to add
    Value* a = newValue(-10, NULL, NO_ANCESTORS, "a");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // ReLU a
    Value* c = ReLU(a, graphStack);

    check_opResultFields(c, a, NULL, 0, 0, reluBackward, "relu", UNARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}

int main(void){
    
    test_newValue();
    test_Add();
    test_Mul();
    test_ReLU();       
}