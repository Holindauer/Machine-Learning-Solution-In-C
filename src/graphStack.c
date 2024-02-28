#include "lib.h"
#include "autoGrad.h"

// graphStack.c

/**
 * @note newGraphStack() allocates memory for a GraphStack struct
 * @return ptr to the new graph stack
*/
GraphStack* newGraphStack(void){

    // Allocate mem 
    GraphStack* stack = (GraphStack*)malloc(sizeof(GraphStack));
    assert(stack != NULL);

    // init w/ one Node that is empty
    stack->head = (GraphNode*)malloc(sizeof(GraphNode));
    assert(stack->head != NULL);

    stack->head->next = NULL;
    stack->head->pValStruct = NULL;
    
    stack->len = 1;

    return stack;
}

/**
 * @note pushGraphStack pushes a new Value onto an existing GraphStack
*/
void pushGraphStack(GraphStack* stack, Value* value){

    assert(stack != NULL);
    assert(value != NULL);

    // allocate mem
    GraphNode* node = (GraphNode*)malloc(sizeof(GraphNode));
    assert(node != NULL);

    // set node fieds
    node->pValStruct = value;
    node->next = stack->head;

    // update stack head
    stack->head = node;
    stack->len++;
}

/**
 * @note popGraphStack() pops a node off the stack
 * @dev popGraph deallocates memory for the Value struct at the head node
*/
void popGraphStack(GraphStack* stack){

    assert(stack != NULL);
    assert(stack->head != NULL);

    // temp ptr to second node in the stack
    GraphNode* next = stack->head->next;

    // retrieve the value struct from the head
    Value* pValStruct = stack->head->pValStruct;
    
    // free head node
    if (pValStruct != NULL){
        freeValue(&pValStruct);
        
    }
    free(stack->head);  

    // adjust the head node
    stack->head = next;
    stack->len--;
}

/**
 * @note releaseGraph() pops (deallocates) all nodes within a graph stack
 * @dev the GraphStack itself is preservered
 * @param ptr to a graphStackStruct to release the Graph of
*/
void releaseGraph(GraphStack* graphStack) {

    while(graphStack->len > 1){
        popGraphStack(graphStack);
    }

    assert(graphStack->head != NULL);
    assert(graphStack->head->pValStruct == NULL);
    assert(graphStack->head->next == NULL);
}

/**
 * @note graphPreservingStackRelease() frees all memory associaed with a graph stack struct, without
 * deallocating the Value structs inside.
 * 
*/
void graphPreservingStackRelease(GraphStack** graphStack){

    // retrieve head node
    GraphNode* graphNode = (*graphStack)->head; 

    // free all graph nodes in the graphStack
    while(graphNode != NULL){

        // save next
        GraphNode* next = graphNode->next;

        free(graphNode);

        // move forward 1 node
        graphNode = next;
    }

    // free graphStack indirect val, set to null
    free(*graphStack);
    *graphStack = NULL;
}   

