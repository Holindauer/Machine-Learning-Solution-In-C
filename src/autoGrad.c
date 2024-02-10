#include "libraries.h"
#include "macros.h"
#include "structs.h"


// autoGrad.c

/**
 * @notice autoGrad.c is where the functionality for reverse mode automatic differentiation is implemented.
 * @notice This implemenation is based on Andrew Karpathy's micrograd. https://github.com/karpathy/micrograd
 * @dev The central struct involved in autoGrad.c is the Value struct, which represents a single value in a directed 
 * acyclic computational graph of data in the nn forward pass. The graph is constructed as operations are performed.
 * These operations are simple enough that their derivatives are easily calculated (adds, multiplies, etc).
 * By chaining together these simple derivatives backwards through the graph, the grad of cost wrt each node can be calculated.
 * @dev Value structs also contain a pointer to a function that is used to calculate the derivative of the operation
 * that produced the value
 * @dev calling the function pointed to will populate the gradient across the entire graph. 
 * @dev Each value struct will also contain an array of pointer the ancestor values that created it. This is how the
 * graph is traversed in reverse to calculate the gradients.
 * 
*/


/**
 * @notice newValue allocates memory for a new Value struct and initializes it with the given value,
 * ancestors, and operation string. It returns a pointer to the new Value struct. 
 * @param _value The scalar/double value of the node
 * @param _ancestors, An array of pointers to the ancestor nodes that created this node
 * @param _ancestorArrLen The length of the ancestor array
 * @param _opStr A string signifying the operation that created this node
 * @return A pointer to the newly created Value struct
*/
Value* newValue(double _value, Value* _ancestors[], int _ancestorArrLen, char _opStr[]){

    // allocate memory for the value struct
    Value* v = (Value*)malloc(sizeof(Value));
    assert(v != NULL);

    // set Value values
    v->value = _value;
    v->grad = 0.0;
    v->refCount = 1; // Starting its own reference count for graph deallocation purposes
    v->ancestorArrLen = _ancestorArrLen;
    v->isMLP = 0; // by default a value is not part of an MLP
 
    // if no ancestors are given, set the ancestors pointer to NULL
    if (_ancestorArrLen == NO_ANCESTORS && _ancestors == NULL) {
        v->ancestors = NULL;
    } else {
        // Allocate memory for the ancestors array of Value pointers into the value struct.
        v->ancestors = (Value**)malloc(_ancestorArrLen * sizeof(Value*)); // Assign directly to v->ancestors.
        assert(v->ancestors != NULL);

        // Copy ancestor nodes into the value struct.
        for (int i = 0; i < _ancestorArrLen; i++) {
            v->ancestors[i] = _ancestors[i];
            _ancestors[i]->refCount++; // Incrementing the ancestor's reference count to reflect that it's being used as an ancestor
        }
    }

    // set Backward function pointer to NULL
    v->Backward = NULL;

    // allocate memory for the operation string and copy into value struct
    v->opStr = (char*)malloc(strlen(_opStr) + 1); 
    assert(v->opStr != NULL);
    strcpy(v->opStr, _opStr); 

    return v;
}


// Forward declaration of the recursive handler to make it available for releaseGraph
void releaseGraphRecursive(Value** v);

// Landing bay function
void releaseGraph(Value** v) {
    if (v == NULL || *v == NULL) {
        return; // Early exit if the input is NULL
    }

    // Call the recursive handler
    releaseGraphRecursive(v); 
}

// Recursive handler
void releaseGraphRecursive(Value** v) {

    // Base case for recursion
    if (*v == NULL) {
        return; 
    }

    // Decrement the refCount and only proceed if it's zero
    if (--(*v)->refCount == 0) {
        if ((*v)->ancestors != NULL) {

            // Recursively release the graph of each ancestor
            for (int i = 0; i < (*v)->ancestorArrLen; i++) {
                if ((*v)->ancestors[i] != NULL) {
                    releaseGraphRecursive(&((*v)->ancestors[i]));
                }
            }

            // only deallocate non mlp structures
            if ((*v)->isMLP == 0) {
                free((*v)->ancestors);
                (*v)->ancestors = NULL;
            }
        }

        // Deallocation of the Value struct itself
        if ((*v)->opStr == 0) {
            // deallocate the operation string
            if ((*v)->opStr != NULL) {
                free((*v)->opStr);
                (*v)->opStr = NULL;
            }
            // deallocate the value struct
            free(*v);
            *v = NULL;
        }
    }
}




/**
 * @notice freeValue() frees the memory allocated for a Value struct and it's ancestors array and operation string
*/
void freeValue(Value* v){

    if (v->ancestors != NULL){
        free(v->ancestors);
    }
    if (v->opStr != NULL){
        free(v->opStr);
    }
    if (v != NULL){
        free(v);
    }
}


/**
 * @notice addBackard() is used within Add() to store the derivative operations of addition in the Value struct
 * @dev calling addBackward() updates the gradients of the ancestor nodes of the input Value structs
 * @dev the partial derivative for this op z = x + y is dz/dx = 1 and dz/dy = 1
*/
void addBackward(Value* v) {

    // Ensure v has at least 2 ancestors since it's the result of addition
    assert(v != NULL && v->ancestors != NULL);
    assert(v->ancestors[0] != NULL && v->ancestors[1] != NULL);

    // Propagate the gradient to both ancestors. Since dz/dx = 1 and dz/dy = 1 for addition, we 
    // simply add v's gradient to each ancestor's gradient because 1 * v->grad == v->grad
    for (int i = 0; i < 2; i++) {      // 2 ancestors for addition
        if (v->ancestors[i] != NULL) { 
            v->ancestors[i]->grad += v->grad; 
        }
    }
}


/**
 * @notice Add() is used to add two Value structs together. It returns a new Value struct that 
 * represents the sum of the two input Value structs.
 * @dev The function also sets the Backward function pointer of the new Value struct to
 *  the address of the addBackward function.
*/
Value* Add(Value* a, Value* b) {
    assert(a != NULL && b != NULL);

    // Create a new Value for the sum
    Value* sumValue = newValue(a->value + b->value, (Value*[]){a, b}, 2, "add");

    // increment ancestor reference counts
    a->refCount++;
    b->refCount++;

    // by default, an added value is not part of an MLP
    sumValue->isMLP = 0;

    // Set the backward function pointer to addBackward
    sumValue->Backward = addBackward;

    return sumValue;
}


/**
 * @notice mulBackward() is used within Mul() to store the derivative operations of multiplication in the Value struct
 * @dev calling mulBackward() updates the gradients of the ancestor nodes of the input Value structs
 * @dev the chain rule for z = x * y is dz/dx = y and dz/dy = x
*/
void mulBackward(Value* v) {
    assert(v != NULL && v->ancestors != NULL);
    assert(v->ancestors[0] != NULL && v->ancestors[1] != NULL);

    // Apply the chain rule for multivariate multiplication: dz/dx = y * dz and dz/dy = x * dz
    Value* x = v->ancestors[0];
    Value* y = v->ancestors[1];
    x->grad += y->value * v->grad; // dz/dx = y
    y->grad += x->value * v->grad; // dz/dy = x
}

/**
 * @notice Mul() is used to multiply two Value structs together. It returns a new Value struct that 
 * represents the product of the two input Value structs.
 * @dev The function also sets the Backward function pointer of the new Value struct to the address of the mulBackward function.
*/
Value* Mul(Value* a, Value* b) {
    assert(a != NULL && b != NULL);

    // Create a new Value for the product
    Value* productValue = newValue(a->value * b->value, (Value*[]){a, b}, 2, "mul");

    // increment ancestor reference counts
    a->refCount++;
    b->refCount++;

    // by default, a multiplied value is not part of an MLP
    productValue->isMLP = 0;

    // Set the backward function pointer to mulBackward
    productValue->Backward = mulBackward;

    return productValue;
}



/**
 * @notice reluBackward() is used within ReLU() to store the derivative operations of the ReLU activation function in the Value struct
 * @dev calling reluBackward() updates the gradients of the ancestor nodes of the input Value structs
 * @dev the chain rule for ReLU is dz/dx = 1 if x > 0 else 0
*/
void reluBackward(Value* v) {
    
    // the only case where ancestors can be null is if we are a node created by newValue()
    if (
        strcmp(v->opStr, "add") == 0 ||
        strcmp(v->opStr, "mul") == 0 ||
        strcmp(v->opStr, "relu") == 0
    ){
        assert(v != NULL && v->ancestors != NULL && v->ancestors[0] != NULL );
    }
    
    // Apply the chain rule for ReLU: dz/dx = 1 if x > 0 else 0
    Value* x = v->ancestors[0];
    if (x->value > 0) {
        x->grad += v->grad; // dz/dx = 1 if x > 0
    }
    // No action needed if x <= 0 since dz/dx = 0 and does not contribute to the gradient
}

/**
 * @notice ReLU() is used to apply the ReLU activation function to a Value struct. It returns a new Value struct that represents 
 * the result of the ReLU activation function applied to the input Value struct.
 * @dev The function also sets the Backward function pointer of the new Value struct to the address of the reluBackward function.
 * @dev The ReLU activation function is defined as f(x) = max(0, x)
 * @dev The derivative of ReLU is 1 if x > 0 else 0
*/
Value* ReLU(Value* a) {
    assert(a != NULL);

    // Create a new Value for the ReLU activation
    double reluResult = a->value > 0 ? a->value : 0; // f(x) = max(0, x)
    Value* reluValue = newValue(reluResult, (Value*[]){a}, 1, "relu");

    // increment ancestor reference counts
    a->refCount++;

    // by default, a ReLU value is not part of an MLP
    reluValue->isMLP = 0;

    // Set the backward function pointer to reluBackward
    reluValue->Backward = reluBackward;

    return reluValue;
}


/**
 * @notice dfs() is a helper function used to perform a depth-first search on the graph of Value structs.
 * @dev dfs() is called within the reverseTopologicalSort() function to recursively visit all nodes in the graph and push 
 * them onto a stack in reverse order.
 * @dev The function uses a hash table to store visited nodes and avoid infinite loops in the graph traversal.
 * @ a depth first search ensures that no node is input to the stack before all of its ancestors have been input.
*/
void dfs(Value* v, HashTable* visitedTable, Value*** stack, int* index) {  

    // If the node has already been visited, return
    if (isVisited(visitedTable, v)) {
        return;
    } 

    // otherwise mark the node as visited in the hash table
    insertVisited(visitedTable, v);

    // Recursive call to visit all ancestors of the current node
    for (int i = 0; v->ancestors != NULL && v->ancestors[i] != NULL; i++) {
        dfs(v->ancestors[i], visitedTable, stack, index);
    }

    // Push the current node onto the stack
    (*stack)[(*index)++] = v;
}


/**
 * @notice reverseArray() is a helper function used to reverse the order of an array of Value structs.
*/
void reverseArray(Value** arr, int start, int end) {
    while (start < end) {
        Value* temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}


/**
 * @notice reverseTopologicalSort() is a helper function used to perform a topological sort on the graph of Value structs.
 * @dev reverseTopologicalSort() is called within the Backward() function to sort a Value's computational graph in topological order
 * @dev The function uses a depth-first search to visit all nodes in the graph and push them onto a stack in reverse order.
 * This ensures that the nodes are sorted in topological order and that no node is visited before its ancestors.
 * @param start is the starting Value node of the graph
 * @param sorted is a pointer to an array of Value pointers that will store the topological sort order of the graph ancestors.
 * @param count is a pointer to an integer that will store the number of nodes sorted.
*/
void reverseTopologicalSort(Value* start, Value*** sorted, int* count) {

    assert(*sorted == NULL);

    // use a hash table to store visited nodes
    HashTable* visited = createHashTable(MAX_GRAPH_SIZE); 
    assert(visited != NULL);

    // Allocate memory for a stack to store the topological sort order of graph ancestors.
    *sorted = (Value**)malloc(MAX_GRAPH_SIZE * sizeof(Value*)); 
    assert(*sorted != NULL);

    // Perform depth-first search to visit all nodes and push them onto the stack.
    int index = 0;
    dfs(start, visited, sorted, &index);

    // Reverse the sorted array to get correct topological ordering.
    reverseArray(*sorted, 0, index - 1);
    *count = index; // Update count to reflect number of nodes sorted.

    freeHashTable(visited);
}

// Function to trigger backward propagation using topological sort.

/**
 * @notice Backward() is the main function used to perform reverse mode automatic differentiation on a Value struct 
 * with respect to all of its ancestors.
 * @dev Backward() is called on the output node of the graph to calculate the gradients of all nodes in the graph.
 * @dev The function uses a topological sort to visit all nodes in the graph in reverse order and calculate their gradients.
 * @dev The function then calls the Backward function of each node to calculate the gradients of their ancestors.
 * @dev The function sets the gradient of the starting node to 1 to begin the backpropagation process.  
 * @dev The function then calls the Backward function of each node to calculate the gradients of their ancestors.
*/
void Backward(Value* v) {

    // Ensure the starting node is not NULL
    assert(v != NULL);

    // declare a pointer for an array of Value pointers
    Value** sorted = NULL;
    int count = 0;

    // Perform a topological sort on the graph of Value structs. This will in place sort
    // fill sorted array with the topological sort order of the graph ancestors.
    reverseTopologicalSort(v, &sorted, &count);

    // Set gradient of the starting node to 1.
    v->grad = 1.0;

    // Process nodes in topologically sorted order.
    for (int i = 0; i < count; i++) {
        
        if (sorted[i]->Backward != NULL) { // Ensure backward function exists
            sorted[i]->Backward(sorted[i]);
        }
    }

    // Clean up
    free(sorted);
}


