#include "libraries.h"
#include "macros.h"


/**
 * @notice gradStructs.h contains struct definitions that are used in the autodiff process
*/


/**
 * @notice pBackwardFunc is a pointer to a function that computes the derivative of the operation 
 * that produced a given Value struct based on it's ancestor values.
 * @param Value* The value struct to compute the derivative for based on it's ancestors.
*/
typedef struct _value Value; // <--- forward declaration for self-reference in pBackwardFunc
typedef void (*pBackwardFunc)(Value*);


/**
 * @notice Value is the central struct in the autoGrad.c implementation. It represents a single value in the
 * computational graph as it passes through the network. The graph is constructed as operations are performed.
 * @param value The value of the node
 * @param grad The gradient of the node
 * @param Backward ptr to a d/dx function for the operation that produced the value
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




