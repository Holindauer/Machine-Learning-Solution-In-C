#pragma once
#include "value.h"
#include "autoGrad.h"

/**
 * @note hashTable.h contains the struct definitions for a hashTable that stores pointers to Value structs
 * @dev this hash table implementation is used in the Backward() funciton of autoGrad.c during a reverse 
 * topological sort of the computational graph.
*/

/**
 * @note BucketNode represents a single node in a bucket within a HashTable 
 * @param key The Value ptr stored at this node
 * @param next A pointer to the next node in the bucket linked list of the HashTable struct
*/
typedef struct bucketNode {
    Value* key;
    struct bucketNode* next;
} BucketNode;

/**
 * @note HashTable represents a hash table for Value struct pointers
 * @param buckets An array of pointers to head ptrs of singly linked lists of BucketNodes
 * @param size The number of buckets in the hash table 
*/
typedef struct {
    BucketNode** buckets;
    int size;
}HashTable;


// HashTable constructor destructor 
HashTable* newHashTable(int size);
void freeHashTable(HashTable** tablePtr);

// HashTable funcs
unsigned int hashValuePtr(Value* value, int tableSize);
BucketNode* newBucketNode(Value* value);
void insertBucketListAtEnd(BucketNode** headNode, Value* value);
void insertHashTable(HashTable* table, Value* value);
int isInHashTable(HashTable* table, Value* value);