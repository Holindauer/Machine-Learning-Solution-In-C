#include "libraries.h"

/**
 * @notice autoGrad.c is where the functionality for reverse mode automatic differentiation is implemented.
 * @notice This implemenation is based on Andrew Karpathy's micrograd. https://github.com/karpathy/micrograd
 * @dev The central struct involced in autoGrad is the Value struct, which represents a single value in the 
 * computation graph as it passes through the network. The graph is constructed as operations are performed.
 * These operations are simple enough that their derivatives can be easily calculated (adds, multiplies, etc).
 * By chaining together these simple derivatives using the graph, the grad of cost wrt each node can be calculated.
 * @dev Value structs also contain a pointer to a function that is used to calculate the derivative of the operation
 * that produced the value. 
 * @dev Each value struct will also contain an array of pointer the ancestor values that created it. This is how the
 * graph is traversed in reverse to calculate the gradients.
*/

/**
 * @notice pBackwardFunc is a pointer to a function that computes the derivative of the operation 
 * that produced a Value struct based on it's ancestor values.
 * @param Value* The value struct to compute the derivative for based on it's ancestors.
*/
typedef struct _value Value; // <--- forward declaration for self-reference in pBackwardFunc
typedef void (*pBackwardFunc)(Value*);


/**
 * @notice Value is the central struct in the autoGrad implementation. It represents a single value in the
 * computation graph as it passes through the network. The graph is constructed by as operations are performed.
 * @param value The value of the node
 * @param grad The gradient of the node
 * @param Backward ptr to d/dx function for the operation that produced the value
 * @param ancestors arr of ancestor nodes (dynamically allocated)
 * @param op Str of operation that produced the value (debugging)
*/
typedef struct _value {
    double value;             
    double grad;              
    pBackwardFunc Backward; 
    Value **ancestors;       
    char* opStr;                 
} Value;


/**
 * @notice newValue allocates memory for a new Value struct and initializes it with the given value,
 * ancestors, and operation string. It returns a pointer to the new Value struct. 
 * @param value The scalae/double value of the node
 * @param ancestors, An array of pointers to the ancestor nodes that created this node
 * @param opStr The operation string that created this node
 * @return A pointer to the newly created Value struct
*/
Value* newValue(double value, Value** ancestors, char opStr[]){

    // allocate memory
    Value* v = (Value*)malloc(sizeof(Value));

    // initizlize the value struct
    v->value = value;
    v->grad = 0;
    v->ancestors = ancestors;

    // allocate memory for the operation string and copy into value struct
    v->opStr = (char*)malloc(strlen(opStr) + 1); 
    strcpy(v->opStr, opStr); 

    return v;
}

