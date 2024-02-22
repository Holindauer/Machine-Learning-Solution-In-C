#include "autoGrad.h"
#include "hashTable.h"

/**
 * @test test_hashValuePtr() tests the hashValuePtr() func, ensuring that the same Value ptr will
 * always be brought to the same uint.
*/
void test_hashValuePtr(void){

    printf("test_hashTable...");

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
 * @test test_hashTableEmptyDestruction() creates a new empty hash table and destructs it
*/
void test_hashTableEmptyDestruction(void){

    printf("test_hashTableEmptyDestruction...");

    // new hash table
    HashTable* hashTable = newHashTable(100);
    assert(hashTable != NULL);

    // destroy hash table
    freeHashTable(&hashTable);
    assert(hashTable == NULL);
   
    printf("PASS!\n");
}


/**
 * @test test_newBucketNode() ensures newBucketNode() inits a BucketNode as expected
*/
void test_newBucketNode(void){

    printf("test_newBucketNode...");

    // create a value to store in the BucketNode
    Value* v = newValue(10, NULL, NO_ANCESTORS, "v");

    // new BN w/ v inside
    BucketNode* bucketNode = newBucketNode(v);

    // validate
    assert(bucketNode != NULL);
    assert(bucketNode->key == v);
    assert(bucketNode->next == NULL);

    // cleanup
    freeValue(&v);
    free(bucketNode);

    printf("PASS!\n");
}


/**
 * @test test_insertBucketListAtEnd() tests the two different insertion cases within insertBucketListAtEnd()
 * 
*/
void test_insertBucketListAtEnd(void){

    printf("test_insertBucketListAtEnd()...");


    // newHashTable() inits all bucket head nodes to NULL for insertBucketListAtEnd() to insert into 
    BucketNode* headNode = NULL;

    // create Value
    Value* v1 = newValue(10, NULL, NO_ANCESTORS, "v1");

    // insert into empty list
    insertBucketListAtEnd(&headNode, v1);

    // validate insertion
    assert(headNode != NULL);
    assert(headNode->key == v1);
    assert(headNode->next == NULL);

    // create another Value
    Value* v2 = newValue(20, NULL, NO_ANCESTORS, "v2");

    // insert into non empty list
    insertBucketListAtEnd(&headNode, v2);

    // validate insertion
    assert(headNode != NULL);
    assert(headNode->key == v1);
    assert(headNode->next != NULL);
    assert(headNode->next->key == v2);
    assert(headNode->next->next == NULL);    

    // cleanup
    freeValue(&v1);
    freeValue(&v2);
    free(headNode->next);
    free(headNode);

    printf("PASS!\n");
}

/**
 * @test test_reinsertionBucketListAtEnd() ensures that when the same value is attempted to be reinserted 
 * into a BucketList, insertBucketListAtEnd() returns early, not linking any new nodes.
*/
void test_reinsertionBucketListAtEnd(void){

    printf("reinsertionBucketListAtEnd...");


    // newHashTable() inits all bucket head nodes to NULL for insertBucketListAtEnd() to insert into 
    BucketNode* headNode = NULL;

    // create Value
    Value* v1 = newValue(10, NULL, NO_ANCESTORS, "v1");

    // insert into empty list
    insertBucketListAtEnd(&headNode, v1);

    // validate insertion
    assert(headNode != NULL);
    assert(headNode->key == v1);
    assert(headNode->next == NULL);

    // insert into empty list
    insertBucketListAtEnd(&headNode, v1);

    // validate there is no new node
    assert(headNode != NULL);
    assert(headNode->key == v1);
    assert(headNode->next == NULL);

    printf("PASS!\n");
}   

/**
 * @test test_insertHashTable() tests the functionality of the insertHashTable() function
*/
void test_insertHashTable(void){

    printf("test_insertHashTable()...");

    // create hash table
    HashTable* hashTable = newHashTable(100);

    // new value
    Value* v1 = newValue(10, NULL, NO_ANCESTORS, "v1");

    // insert into hash table
    insertHashTable(hashTable, v1);

    // manually get idx of v1
    unsigned int idx = hashValuePtr(v1, 100);

    // retrieve value from hash Table
    Value* internalValue = hashTable->buckets[idx]->key;

    // validate insertion
    assert(internalValue == v1);

    // cleanup 
    freeHashTable(&hashTable);  
    assert(hashTable == NULL);

    printf("PASS!\n");
}

/**
 * @test test_isInHashTable() tests the isInHashTable() function for a value that is both in and a value that 
 * is not in the hash table
*/
void test_isInHashTable(void){

    printf("test_isInHashTable()...");

    // create hash table
    HashTable* hashTable = newHashTable(100);

    // new value
    Value* v1 = newValue(10, NULL, NO_ANCESTORS, "v1");

    // insert into hash table
    insertHashTable(hashTable, v1);

    // check that value is in hash table
    assert(isInHashTable(hashTable, v1) == 1);

    // new value that is not in the hash table
    Value* v2 = newValue(20, NULL, NO_ANCESTORS, "v2");

    // check value is not in hash table
    assert(isInHashTable(hashTable, v2) == 0);

    // cleanup 
    freeHashTable(&hashTable);  
    assert(hashTable == NULL);

    printf("PASS!\n");
}


int main(void){

    test_hashValuePtr();
    test_hashTableEmptyDestruction();
    test_newBucketNode();
    test_insertBucketListAtEnd();
    test_reinsertionBucketListAtEnd();
    test_insertHashTable();
    test_isInHashTable();

    return 0;
}