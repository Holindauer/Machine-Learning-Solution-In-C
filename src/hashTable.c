#include "libraries.h"
#include "macros.h"


/**
 * @notice hashTable.c c implements a hashTable data structure that is used within the autoGrad.c file 
 * during the reverseToplogicalSort function. This is necessary to avoid infinite loops in the graph traversal.
 * @dev The hash table is used specifically,to map Nodes to a unique integer value
*/


/**
 * @notice VisitedNode is a struct that represents a node in the hash table
 * @param key The value of the node
 * @param next A pointer to the next node in the list
*/
typedef struct VisitedNode {
    Value* key;
    struct VisitedNode* next;
} VisitedNode;

/**
 * @notice HashTable is a struct that represents a hash table
 * @param buckets An array of pointers to VisitedNode structs
 * @param size The size of the hash table
*/
typedef struct {
    VisitedNode** buckets;
    int size;
} HashTable;


/**
 * @notice hashValuePtr() is a helper function that takes in a pointer and a size and returns a unique integer value
 * @dev The calculation ptr >> 3 is a bit shift operation that is used to convert the pointer to an integer value.
 * modding by the size ensures that the value is within the range of the hash table
*/
unsigned int hashValuePtr(void* ptr, int size) {
    return ((unsigned long)ptr >> 3) % size;
}

/**
 * @notice createHashTable() takes a size and allocates memory for and returns a pointer to a HashTable 
*/
HashTable* createHashTable(int size) {

    // Allocate memory for the table
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    assert(table != NULL);

    // set table size
    table->size = size;

    // Allocate memory for the buckets
    table->buckets = (VisitedNode**)malloc(sizeof(VisitedNode*) * size);
    assert(table->buckets != NULL);

    // initialize the buckets to NULL
    for (int i = 0; i < size; i++) {
        table->buckets[i] = NULL;
    }
    return table;
}

/**
 * @notice insertVisited() takes a hash table and a value and inserts the value into the hash table
 * @dev The function first hashes the value and then inserts it into the hash table
*/
void insertVisited(HashTable* table, Value* value) {

    // hash the value
    unsigned int index = hashValuePtr(value, table->size);

    // allocate memory for the new node
    VisitedNode* node = (VisitedNode*)malloc(sizeof(VisitedNode));
    assert(node != NULL);

    // set node's key
    node->key = value;

    // set node's next pointer
    node->next = table->buckets[index]; 

    // insert the node into the hash table at the hashed index
    table->buckets[index] = node; 
}


/**
 * @notice isVisited() takes a hash table and a value and returns true if the value is in the hash table
 * 
*/
bool isVisited(HashTable* table, Value* value) {

    // hash the value
    unsigned int index = hashValuePtr(value, table->size);

    // get the node at the hashed index
    VisitedNode* node = table->buckets[index];

    // iterate through the list
    while (node != NULL) {  

        // return true if the value is in the list
        if (node->key == value) {
            return true;
        }
        node = node->next;
    }
    return false;
}


/**
 * @notice freeHashTable() takes a hash table and frees the memory allocated for the hash table
 * @dev The function first frees the memory allocated for each node in the hash table and then frees the memory allocated for the hash table itself
*/
void freeHashTable(HashTable* table) {

    // free the memory allocated for each node in the hash table
    for (int i = 0; i < table->size; i++) {


        VisitedNode* node = table->buckets[i];
        while (node != NULL) { // iterate through the list
            VisitedNode* temp = node; 
            node = node->next;
            free(temp);
        }
    }
    free(table->buckets);
    free(table);
}
