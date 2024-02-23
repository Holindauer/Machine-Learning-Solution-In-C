#include "lib.h"

/**
 * @test test_newGraphStack() checks whether the a graph stack is properly initialized by the
 * newGraphStack function in graphStack.c
*/
void test_newGraphStack(void){

    printf("test_newGraphStack...");

    // new GraphStack
    GraphStack* graphStack = newGraphStack();

    // check init
    assert(graphStack != NULL);
    assert(graphStack->len == 1);
    assert(graphStack->head != NULL);
    assert(graphStack->head-> next == NULL);
    assert(graphStack->head->pValStruct == NULL);
    
    // cleanup
    free(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_pushGraphStack() tests that the push mechanism for graph stacks is working as expected
*/
void test_pushGraphStack(void){

    printf("test_pushGraphStack...");

    // init some Values to push to the stack
    Value* v1 = newValue(1, NULL, NO_ANCESTORS, "v1");
    Value* v2 = newValue(2, NULL, NO_ANCESTORS, "v2");
    Value* v3 = newValue(3, NULL, NO_ANCESTORS, "v3");

    // init GraphStack
    GraphStack* graphStack = newGraphStack();

    // push to the stack
    pushGraphStack(graphStack, v1);
    pushGraphStack(graphStack, v2);
    pushGraphStack(graphStack, v3);

    // validate pushes for head 
    assert(graphStack->head->pValStruct->value == 3); 
    assert(strcmp(graphStack->head->pValStruct->opString, "v3") == 0);

    // validate next node
    assert(graphStack->head->next->pValStruct->value == 2);
    assert(strcmp(graphStack->head->next->pValStruct->opString, "v2") == 0);

    // validate initial node
    assert(graphStack->head->next->next->pValStruct->value == 1);
    assert(strcmp(graphStack->head->next->next->pValStruct->opString, "v1") == 0);

    // cleanup
    freeValue(&v1);
    freeValue(&v2);
    freeValue(&v3);
    free(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_popGraphStack() tests that the pop mechanism for graph stacks is working correctly
*/
void test_popGraphStack(void){

    printf("test_popGraphStack...");

    // init some Values to push to the stack
    Value* v1 = newValue(1, NULL, NO_ANCESTORS, "v1");
    Value* v2 = newValue(2, NULL, NO_ANCESTORS, "v2");
    Value* v3 = newValue(3, NULL, NO_ANCESTORS, "v3");

    // init GraphStack
    GraphStack* graphStack = newGraphStack();

    // push to the stack
    pushGraphStack(graphStack, v1);
    pushGraphStack(graphStack, v2);
    pushGraphStack(graphStack, v3);

    // check head node is last push
    assert(v3 == graphStack->head->pValStruct);

    // pop nodes and validate release
    popGraphStack(graphStack);

    // check head node is second to last push now after pop
    assert(graphStack->head->pValStruct == v2);

    // pop again
    popGraphStack(graphStack);

    assert(graphStack->head->pValStruct == v1);

    printf("PASS!\n");
}


/**
 * @test test_releaseGraph() tests the sequential deallocation of the graph within a graphStack
*/
void test_releaseGraph(void){

    printf("test_releaseGraph...");

    // init some Values to push to the stack
    Value* v1 = newValue(1, NULL, NO_ANCESTORS, "v1");
    Value* v2 = newValue(2, NULL, NO_ANCESTORS, "v2");
    Value* v3 = newValue(3, NULL, NO_ANCESTORS, "v3");
    Value* v4 = newValue(4, NULL, NO_ANCESTORS, "v4");
    Value* v5 = newValue(5, NULL, NO_ANCESTORS, "v5");
    Value* v6 = newValue(6, NULL, NO_ANCESTORS, "v6");

    // init GraphStack
    GraphStack* graphStack = newGraphStack();

    // push to the stack
    pushGraphStack(graphStack, v1);
    pushGraphStack(graphStack, v2);
    pushGraphStack(graphStack, v3);
    pushGraphStack(graphStack, v4);
    pushGraphStack(graphStack, v5);
    pushGraphStack(graphStack, v6);

    // check head node is last push
    assert(v6 == graphStack->head->pValStruct);
    assert(graphStack->len == 7);

    // release the graph
    releaseGraph(graphStack);

    assert(graphStack-> head != NULL);
    assert(graphStack->head->next == NULL);
    assert(graphStack->head->pValStruct == NULL);


    printf("PASS!\n");
}

/**
 * @test test_reverseGraphStack() checks to make sure the reverseGraphStack() func works as expected
*/
void test_reverseGraphStack(void){

    printf("test_reverseGraphStack()...");

    // init some Values to push to the stack
    Value* v1 = newValue(1, NULL, NO_ANCESTORS, "v1");
    Value* v2 = newValue(2, NULL, NO_ANCESTORS, "v2");
    Value* v3 = newValue(3, NULL, NO_ANCESTORS, "v3");
    Value* v4 = newValue(4, NULL, NO_ANCESTORS, "v4");
    Value* v5 = newValue(5, NULL, NO_ANCESTORS, "v5");
    Value* v6 = newValue(6, NULL, NO_ANCESTORS, "v6");

    // init GraphStack
    GraphStack* graphStack = newGraphStack();

    // push to the stack
    pushGraphStack(graphStack, v1);
    pushGraphStack(graphStack, v2);
    pushGraphStack(graphStack, v3);
    pushGraphStack(graphStack, v4);
    pushGraphStack(graphStack, v5);
    pushGraphStack(graphStack, v6);

    // check stack order is correct
    assert(graphStack->len == 7);
    assert(v6 == graphStack->head->pValStruct);
    assert(v5 == graphStack->head->next->pValStruct);
    assert(v4 == graphStack->head->next->next->pValStruct);
    assert(v3 == graphStack->head->next->next->next->pValStruct);
    assert(v2 == graphStack->head->next->next->next->next->pValStruct);
    assert(v1 == graphStack->head->next->next->next->next->next->pValStruct);
    
    // reverse stack
    reverseGraphStack(&graphStack);

    // check stack order is reversed
    assert(graphStack->len == 7);
    assert(v1 == graphStack->head->pValStruct);
    assert(v2 == graphStack->head->next->pValStruct);
    assert(v3 == graphStack->head->next->next->pValStruct);
    assert(v4 == graphStack->head->next->next->next->pValStruct);
    assert(v5 == graphStack->head->next->next->next->next->pValStruct);
    assert(v6 == graphStack->head->next->next->next->next->next->pValStruct);
    

    // release the graph
    releaseGraph(graphStack);
    assert(graphStack-> head != NULL);
    assert(graphStack->head->next == NULL);
    assert(graphStack->head->pValStruct == NULL);


    printf("PASS!\n");
}


int main(void){

    test_newGraphStack();
    test_pushGraphStack();
    test_popGraphStack();
    test_releaseGraph();
    test_reverseGraphStack();

    return 0;
}