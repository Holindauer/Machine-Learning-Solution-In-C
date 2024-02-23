#pragma once

/**
 * @notice pBackwardFunc is a pointer to a function that computes the derivative of the operation 
 * that produced a given Value struct based on it's ancestor values.
 * @param Value* The value struct to compute the derivative for based on it's ancestors.
*/
typedef struct _value Value; // <--- forward declaration for self-reference in pBackwardFunc
typedef void (*pBackwardFunc)(Value*);

/**
 * @notice Value represents a single node in the computational graph as it passes through the network. 
 * The graph is constructed as operations are performed.
 * @dev operations are 
 * @param value The double value of the node
 * @param grad The partial derivative value for the node wrt the final output of the graph
 * @param Backward ptr to a d/dx function for the operation that produced the value [Add(), Mull(), ReLU()...]
 * @param ancestors arr of ancestor nodes (dynamically allocated) 
 * @param op String indicating the operation that produced the value (debugging)
 * @param ancestorArrLen length of the ancestors array
*/
typedef struct _value {
    double value;             
    double grad;              
    pBackwardFunc Backward; 
    Value** ancestors;
    char* opString; 
    int ancestorArrLen;        
} Value;