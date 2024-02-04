#include "libraries.h"

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
 * @test test_insertVisited() tests the insertVisited function by inserting a new
 * value into the hash table and checking that it was inserted correctly
 * @dev The function creates a new value node, inserts it into the hash table, and then checks that it was inserted correctly
*/
void test_insertVisited(void){

    // create a new hash table
    HashTable* table = createHashTable(10);

    // create a new value node
    Value* v = newValue(10, NULL, NO_ANCESTORS, "v");
    Value* v3 = newValue(30, NULL, NO_ANCESTORS, "v3");

    // insert the value into the hash table
    insertVisited(table, v);
    insertVisited(table, v3);

    // check that the value was inserted correctly
    assert(table->buckets[hashValuePtr(v, table->size)]->key == v);
    assert(table->buckets[hashValuePtr(v3, table->size)]->key == v3);

    freeHashTable(table);
    freeValue(v);
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
