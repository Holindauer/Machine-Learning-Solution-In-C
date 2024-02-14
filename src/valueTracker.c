#include "libraries.h"
#include "macros.h"
#include "structs.h"

/**
 * @notice valueTracker.c contains an implementation of a stack used to track newly created Value structs.
 * The goal here is to be able to divorce the deallocation of the computational graph from the need to 
 * traverse the graph while doing so. 
 * @dev This mechanism is used to avoid recursively deallocating the graph, which is less obvious and more 
 * error prone due to potential double frees from acestors being in multiple places in the graph.
 * 
*/



