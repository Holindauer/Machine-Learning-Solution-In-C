#include "libraries.h"
#include "macros.h"
#include "structs.h"

/**
 * @notice graphTracker.c implements a stack used to track newly created Value structs. The goal here is to divorce 
 * the deallocation of the computational graph from the need to traverse the graph while doing so. 
 * @dev This mechanism is used to avoid recursively deallocating the graph, which is less obvious and more 
 * error prone due to potential double frees from acestors being in multiple places in the graph. 
 * @dev by pushing to a stack at the time of creation, we ensure that all nodes in the stack are unique and 
 * will not cause a double free when releasing memory.
*/


/**
 * @notice newGraphStack() allocates memory for a new GraphStack and returns a pointer to it
*/
GraphStack* newGraphStack(void){

    // allocate memory for the stack
    GraphStack* stack = (GraphStack*)malloc(sizeof(GraphStack));
    assert(stack != NULL);

    // init stack members
    stack->head = NULL;
    stack->len = 0;

    return stack;
}

/**
 * @notice pushGraphStack() pushes a Value struct onto the stack
 * 
*/
void pushGraphStack(GraphStack* stack, Value* value){

    // allocate memory for the new node
    GraphNode* node = (GraphNode*)malloc(sizeof(GraphNode));
    assert(node != NULL);

    // set the node's value to the value passed in
    node->value = value;

    // set the node's next pointer to the current head of the stack
    node->next = stack->head;

    // set the head of the stack to the new node
    stack->head = node;

    // increment the length of the stack
    stack->len++;
}

/**
 * @notice popGraphStack() pops a Value struct off the stack
*/
void popGraphStack(GraphStack* stack){

    // if the stack is empty, return
    if(stack->head == NULL){
        return;
    }

    // set the head of the stack to the next node
    GraphNode* next = stack->head->next;

    // free the head of the stack
    freeValue(stack->head->value);
    free(stack->head);

    // set the head of the stack to the next node
    stack->head = next;

    // decrement the length of the stack
    stack->len--;
}



/**
 * @notice releaseGraph() is used to deallocate all memory associated with a graph that has been
 * stored up in the GraphStack
*/
void releaseGraph(GraphStack* graphStack) {
    // pop all nodes off the stack
    for (int i = 0; i < graphStack->len; i++){
        popGraphStack(graphStack);
    }
}