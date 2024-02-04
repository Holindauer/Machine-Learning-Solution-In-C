#include "libraries.h"
#include "macros.h"
// #include "../src/autoGrad.c"
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
 * @test test_insertVisited_and_chaining tests the insertVisited function by inserting two values into the hash table
 * and checking that they were inserted correctly
 * @dev This test also checks that chaining is working correctly
 * @dev Chaining is the process of handling collisions in a hash table by creating a linked list of nodes at each index
 * @dev This test assumes that the two values will hash to the same index
 * @dev This test also checks the isVisited function 
*/
void test_insertVisited_and_chaining(void) {
    // Create a new hash table
    HashTable* table = createHashTable(10);

    // Create new value nodes
    // Intentionally choosing values likely to hash to the same bucket to test chaining
    Value* v1 = newValue(10, NULL, NO_ANCESTORS, "v1");
    Value* v2 = newValue(30, NULL, NO_ANCESTORS, "v2");

    // Insert the values into the hash table
    insertVisited(table, v1);
    insertVisited(table, v2);

    // Calculate hash indices
    unsigned int index1 = hashValuePtr(v1, table->size);
    unsigned int index2 = hashValuePtr(v2, table->size);

    // Check if the values are inserted and if chaining handles collisions correctly
    assert(index1 == index2); // Assuming collision for this test case; adjust as needed

    // Verify v2 is at the head of the list (most recently inserted)
    assert(table->buckets[index2]->key == v2);

    // Verify v1 is the next node in the list
    assert(table->buckets[index1]->next->key == v1);

    // Verify that the list ends here
    assert(table->buckets[index1]->next->next == NULL);

    // Test retrieval functionality (if isVisited works as expected)
    assert(isVisited(table, v1) == true);
    assert(isVisited(table, v2) == true);

    // Cleanup
    freeHashTable(table);
    freeValue(v1);
    freeValue(v2);
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

    printf("\nRunning Hash Table Tests...\n");

    // hashTable tests
    test_createHashTable();
    test_insertVisited_and_chaining();
    test_isVisited();

    printf("All Hash Table tests passed!\n");

    return 0;
}