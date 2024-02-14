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

    for (int i = 0; i < 2; i++){
        popGraphStack(stack);
    }

    freeValue(value1);
    freeValue(value2);
    free(stack);

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

    freeValue(value1);
    freeValue(value2);
    free(stack);

    printf("test_popGraphStack passed\n");
}



int main(void){

    printf("\nTesting graphTracker.c\n");

    test_GraphStackInit();
    test_pushGraphStack();
    test_popGraphStack();

    printf("\nAll tests passed!\n");

    return 0;
}