#pragma once
#include "value.h"
#include "autoGrad.h"

// graphStack.h

/**
 * @notice GraphNode is used to track each node within a stack of Value structs (GraphStack). 
 * @dev Each time a Value is created during the forward pass, it is stored in a GraphNode and 
 * pushed to the GraphStack. This allows for sequential deallocation of memory for complicated
 * computation graphs without risk of double frees. 
 * @param pValStruct A pointer to the Value struct at this node
 * @param next A pointer to the next ValueTracker struct in the stack
*/
typedef struct _graphNode {
    Value* pValStruct;
    struct _graphNode* next;
} GraphNode;

/**
 * @notice GraphStack is a stack of GraphNode for storing all Value structs created during the 
 * forward pass.
 * @dev This allows for sequential deallocation of the computational graph.
 * @param head A pointer to the head of the stack
 * @param len The length of the stack
*/
typedef struct {
    GraphNode* head;
    int len;
} GraphStack;

// GraphStack functions
GraphStack* newGraphStack(void);
void pushGraphStack(GraphStack* stack, Value* value);
void popGraphStack(GraphStack* stack);
void releaseGraph(GraphStack* graphStack);
void graphPreservingStackRelease(GraphStack** graphStack);