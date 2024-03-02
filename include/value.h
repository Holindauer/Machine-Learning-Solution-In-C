#pragma once

// value.h


typedef struct _value Value; // <--- forward declaration for self-reference in pBackwardFunc

/**
 * @note pBackwardFunc_Singluar is a pointer to a function that computes the derivative of the operation 
 * that produced a given Value struct based on it's ancestor values.
 * @note pBackwardFunc_Singluar is intended specifically for partial derivatives that can be computed using
 * only the value in the Value ptr and its immediate ancestors (Add(), Mul(), ReLU(), Exp(), Div())
 * @param Value* The value struct to compute the derivative for based on it's ancestors.
*/
typedef void (*pBackwardFunc_Singluar)(Value*);

/**
 * @pBackwardFunc_Loss is a pointer to a function that computes the derivative of the operation that produced
 * the loss for an mlp. This function ptr was implemented specificly for giving categoricalCrossEntropyBackward()
 * access to the extra fields it needs to compute its derivative (target labels and softmax probabilitys, which 
 * are not directly accessible via the loss output's ancestors)
 * @param Value* the output of categoricalCrossEntropy()
 * @param double* an array of softmax probabilities
 * @param Value** array Value struct ptrs of target class labels 
 * @param int length of the above two arrays
*/
typedef void (*pBackwardFunc_Loss)(Value*, double*, Value**, int);



/**
 * @notice Value represents a single node in the computational graph as it passes through the network. 
 * The graph is constructed as operations are performed.
 * @dev operations are 
 * @param value The double value of the node
 * @param grad The partial derivative value for the node wrt the final output of the graph
 * @param Backward ptr to a d/dx function for the operation that produced the value [Add(), Mull(), ReLU()...]
 * @param BackwardLoss ptr to a d/dx function for computing the gradient of the loss (categorical cross entropy)
 * @param ancestors arr of ancestor nodes (dynamically allocated) 
 * @param op String indicating the operation that produced the value (debugging)
 * @param ancestorArrLen length of the ancestors array
*/
typedef struct _value {
    double value;             
    double grad;              
    pBackwardFunc_Singluar Backward; 
    pBackwardFunc_Loss BackwardLoss;
    Value** ancestors;
    char* opString; 
    int ancestorArrLen;        
} Value;