#include "libraries.h"
#include "test_autoGrad.c"
#include "test_hashTable.c"


int main(void) {

  printf("Running tests...\n");

  // autoGrad tests
  test_newValue();
  test_valueOperations();
  test_AddDiff();
  test_MulDiff();
  test_reluDiff();
  // test_Backprop(); currently not working

  // hashTable tests
  test_createHashTable();
  test_insertVisited();
  test_isVisited();

  printf("All tests passed!\n");

  return 0;
}