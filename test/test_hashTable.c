#include "autoGrad.h"
#include "hashTable.h"

/**
 * @test test_hashValuePtr() tests the hashValuePtr() func, ensuring that the same Value ptr will
 * always be brought to the same uint.
*/
void test_hashValuePtr(void){

    printf("test_hashTable.c...");

    Value* v;   

    // test on 150 different new values
    for (int i=0; i<150; i++){

        v = newValue(i, NULL, NO_ANCESTORS, "test");

        unsigned int hash = hashValuePtr(v, 140); // 140 tableSize set arbitrarily

        // hash each value 100 times
        for (int j =0; j<100; j++){
            unsigned int hashCompare = hashValuePtr(v, 140); 
            assert(hash == hashCompare);
        }

        freeValue(&v);
        assert(v == NULL);
    }

    printf("PASS!\n");
}


/**
 * test_hashTableEmptyDestruction() creates a new empty hash table and destructs it
*/
void test_hashTableEmptyDestruction(void){

    printf("test_hashTableEmptyDestruction...");

    HashTable* hashTable = newHashTable(100);
    assert(hashTable != NULL);

    freeHashTable(&hashTable);
    assert(hashTable == NULL);
    

    printf("PASS!\n");
}

int main(void){

    test_hashValuePtr();
    test_hashTableEmptyDestruction();


    return 0;
}