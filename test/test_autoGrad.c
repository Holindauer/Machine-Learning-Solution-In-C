#include "libraries.h"
#include "../src/autoGrad.c"


/**
 * @notice test_newValue tests the newValue function by creating a new value node and checking that it was initialized correctly
 * @dev The function creates some ancestor nodes, creates an array of ancestor nodes, and then creates a new value node.
 * @dev The function then checks that the value was initialized correctly.
*/
int test_newValue(){

    // create some ancestor nodes
    Value* a = newValue(3, NULL, "a");
    Value* b = newValue(4, NULL, "b");
    Value* c = newValue(5, NULL, "c");

    // create an array of ancestor nodes
    Value* ancestors[3] = {a, b, c};

    // create a new value node
    Value* v = newValue(10, ancestors, "add");

    // check that the value was initialized correctly
    assert(v->value == 10);
    assert(v->grad == 0);
    assert(v->ancestors[0] == a);
    assert(v->ancestors[1] == b);
    assert(v->ancestors[2] == c);
    assert(strcmp(v->opStr, "add") == 0);

    free(a->opStr);
    free(a);
    free(b->opStr);
    free(b);
    free(c->opStr);
    free(c);
    free(v->opStr);
    free(v);

    return 0;
}
