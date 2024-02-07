#include "libraries.h"
#include "macros.h"
#include "structs.h"


/**
 * @notice test_hashTable.c contains tests of the hash table defined in hashTable.c
 * @dev this has table is used to avoid infinite loops in the graph traversal during
 * the reverseTopologicalSort function in autoGrad.c 
*/

/**
 * @test test_createHashTable tests the createHashTable function by creating 
 * a new hash table and checking that it was initialized correctly
*/
void test_createHashTable(void){

    // create a new hash table
    HashTable* table = createHashTable(10);

    // check that the table was initialized correctly
    assert(table->size == 10);
    for(int i = 0; i < 10; i++){
        assert(table->buckets[i] == NULL);
    }

    freeHashTable(table);
}


/**
 * @test test_isVisited() tests the isVisited function by inserting a value into the hash table
 * and then checking that it was inserted correctly
*/
void test_isVisited(void){

    // create a new hash table
    HashTable* table = createHashTable(10);

    // create a new value node
    Value* v = newValue(10, NULL, NO_ANCESTORS, "v");

    // insert the value into the hash table
    insertVisited(table, v);

    // check that the value was inserted correctly
    assert(isVisited(table, v) == 1);

    freeHashTable(table);
    freeValue(v);
}


// run the tests
int main(void){

    printf("Running Hash Table Tests...\n");

    // hashTable tests
    test_createHashTable();
    test_isVisited();

    printf("All Hash Table tests passed!\n\n");

    return 0;
}