#include "libraries.h"
#include "test_autoGrad.c"


int main(void) {

  printf("Running tests...\n");

  test_newValue();
  
  printf("All tests passed!\n");

  return 0;
}