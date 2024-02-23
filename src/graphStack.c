#include "lib.h"
#include "autoGrad.h"

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
 * @note reverseGraphStack reverses a GraphStack struct by creating a new stack, then iterating the push of pops to it 
 * @param stack ptr to a ptr of the GraphStack to reverse
*/
void reverseGraphStack(GraphStack** stack){
    assert(stack != NULL);
    assert((*stack) != NULL);

    // create new graph stack to hold reversed stack
    GraphStack* reverseStack = newGraphStack();

    // retrieve head node from stack
    GraphNode* graphNode = (*stack)->head;
    GraphNode* nextGraphNode = NULL;

    // push each element of reverse stack onto new stack
    while(graphNode != NULL && graphNode->pValStruct != NULL){

        // push next val in input param stack to new reverse stack
        pushGraphStack(reverseStack, graphNode->pValStruct);

        // move forward one node, freeing previous (Value ptrs still intact)
        nextGraphNode = graphNode->next;
        free(graphNode);
        graphNode = nextGraphNode;
    }

    // free the input param GraphStack and replace w/ the new reversed version
    free(*stack);
    *stack = reverseStack;
}

/**
 * @note releaseGraph() pops (deallocates) all nodes within a graph stack
 * @dev the GraphStack itself is preservered
*/
void releaseGraph(GraphStack* graphStack) {

    while(graphStack->len > 1){
        popGraphStack(graphStack);
    }

    assert(graphStack->head != NULL);
    assert(graphStack->head->pValStruct == NULL);
    assert(graphStack->head->next == NULL);
}