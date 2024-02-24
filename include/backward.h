#pragma once
#include "value.h"
#include "hashTable.h"
#include "graphStack.h"

// backward.h

// Backpropagation functions
void depthFirstSearch(Value* value, HashTable* visitedHashTable, GraphStack* sortedStack);
void reverseTopologicalSort(Value* start, GraphStack* sortedStack);
void Backward(Value* value);