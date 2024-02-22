#include "autoGrad.h"
#include "hashTable.h"
#include "lib.h"


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

    // Allocate mem for array of each bucket's head ptr
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
    assert(*tablePtr != NULL);

    // free all nodes in all buckets
    for (int i=0; i < (*tablePtr)->size; i++){

        // retrieve head node of i'th bucket
        BucketNode* bucketNode = (*tablePtr)->buckets[i];

        // free each node in the bucket list
        while( bucketNode != NULL){

            // save current node for freeing
            BucketNode* nodeToFree = bucketNode;

            // progress bucketNode to next node
            bucketNode = bucketNode->next;

            if (nodeToFree != NULL){
                free(nodeToFree);
            }        
        }

        // set i'th bucket to NULL
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


// ---------------------------------------------------------------------------------------------------------------------- HashTable Operations

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

/**
 * @note newBucketNode() allocates memory for a new bucket node, placing a Value ptr inside before returning
 * @param value is the Value struct ptr to place inside of the new BucketNode
*/
BucketNode* newBucketNode(Value* value){
    assert(value != NULL);

    // allocate memory for new BucketNode
    BucketNode* newNode = (BucketNode*)malloc(sizeof(BucketNode));
    assert(newNode != NULL);

    // set newNode fields
    newNode->next = NULL;
    newNode->key = value;

    return newNode;
}


/**
 * @note insertBucketListAtEnd() inserts a new BucketNode containing a Valuer struct pointer into one of the bucket 
 * linked lists within a HashTable.
 * @dev We are inserting at the end in order to check if there is already a BucketNode with the same Value ptr as 
 * is in the newNode. If there is such a Value ptr already within the list, the function will return early. Otherwise
 * it will reach the end of the list, allocate memory for a new node, and insert at the end of the list.
 * @param headNode is a ptr to the head node ptr of the bucket to insert into (at the index of the HashTable 
 * bucket array determined by the hashValuePtr() function).
 * @param value the Value struct ptr to store in a newNode and insert into 
*/
void insertBucketListAtEnd(BucketNode** headNode, Value* value){
    assert(headNode != NULL);

    // empty bucket list case
    if (*headNode == NULL){

        // put value into a new BucketNode
        BucketNode* newNode = newBucketNode(value);

        // set new node as the head of the empty list
        *headNode = newNode;

    } else{ // Non empty bucket list case

        BucketNode* node = *headNode;

        // iterate bucket list
        while (node != NULL){
            
            // return early if value alread in the list
            if (node->key == value){
                return; // @note  if returing, no mem was allocated
            }

            // insert once end of list is found
            if (node->next == NULL){
                
                // put value into a new BucketNode
                BucketNode* newNode = newBucketNode(value);

                // insert at end
                node->next = newNode;
            }

            node = node->next;
        }
    }
}

/**
 * @note insertHashTable() applies the hashValuePtr() function to a Value ptr and inserts it into a Hash Table Struct.
 * @dev insertBucketList() is used to insert new bucketNodes into the appropriate bucket linked list of the hash table.
 * @param table ptr to the hash table to insert into
 * @param value the Value struct ptr to insert
*/
void insertHashTable(HashTable* table, Value* value){
    assert(table != NULL && value != NULL);

    // hash value
    unsigned int bucketIndex = hashValuePtr(value, table->size);

    // get address of the ptr to the head node of the bucket list at the Value ptr's bucket index
    BucketNode** headNode = &(table->buckets[bucketIndex]);

    // insert newNode into the retrieved bucket list 
    insertBucketListAtEnd(headNode, value);
}


/**
 * @not isInHashTable() returns 1 or 0 for whether a value struct is currently within a HashTable
 * @param table a HashTable to check 
 * @param value value to check if inside table
*/
int isInHashTable(HashTable* table, Value* value){
    assert(table != NULL && value != NULL);

    // hash value
    unsigned int bucketIndex = hashValuePtr(value, table->size);

    // get address of ptr to head node of the bucket list at Value ptr's bucket index
    BucketNode* node = table->buckets[bucketIndex];

    // traverse list to search for ptr
    while (node != NULL){

        // value found
        if (node->key == value){
            return 1;
        }
        node = node->next;
    }

    // value not found
    return 0;
}

