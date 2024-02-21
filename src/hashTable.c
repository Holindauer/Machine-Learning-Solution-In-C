#include "autoGrad.h"
#include "hashTable.h"
#include "lib.h"


/**
 * ! Hash Table is a bit of a wildcard atm. It will need a good deal of testing
 * 
*/


//---------------------------------------------------------------------------------------------------------------------- HashTable Constructor

/**
 * @note newHashTable() creates a new hash table struct given a size
*/
HashTable* newHashTable(int size){
    assert(size > 0);

    // allocate mem for table
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    assert(table != NULL);

    // set table size
    table->size = size;

    // Allocate mem for buckets
    table->buckets = (BucketNode**)malloc(sizeof(BucketNode*) * size);
    assert(table->buckets != NULL);

    // init each bucket to NULL
    for (int i = 0; i < size; i++){
        table->buckets[i] = NULL;
    }

    return table;
}


//---------------------------------------------------------------------------------------------------------------------- HashTable Destructor

/**
 * @note freeHashTable() frees memory allocated for a HashTable 
 * @dev memory for Value Structs within the HashTable is not deallocated
 * @param tablePtr is a pointer to a pointer to a HashTable struct
*/
void freeHashTable(HashTable** tablePtr){
    assert(tablePtr != NULL);

    // free all nodes in all buckets
    for (int i=0; i< (*tablePtr)->size; i++){

        // retrieve head node of i'th bucket
        BucketNode* bucketNode = (*tablePtr)->buckets[i];

        // free each node in the bucket list
        while( bucketNode != NULL){

            BucketNode* nodeToFree = bucketNode;
            bucketNode = bucketNode->next;

            if (nodeToFree != NULL){
                free(nodeToFree);
            }
        }

        // set i'th bucker to NULL
        (*tablePtr)->buckets[i] = NULL;
    }

    // free buckets array ptr
    if ((*tablePtr)->buckets != NULL){
        free((*tablePtr)->buckets);
    }

    // free table and set ptr to NULL
    if ((*tablePtr) != NULL){
        free(*tablePtr);
        *tablePtr = NULL;
    }
}


//---------------------------------------------------------------------------------------------------------------------- HashTable Operations

/**
 * @note hashValuePtr accepts a pointer to a Value struct and hashes it to an integer value
 * @param value is the Value struct ptr to hash
 * @param tableSize is the number of buckets in the hash table
*/
unsigned int hashValuePtr(Value* value, int tableSize){
    assert(value != NULL && tableSize > 0);

    // cast to long uint, bit shift 3, mod by tableSize to being within [0, tableSize]
    return ((unsigned long)value >> 3) % tableSize;
}
