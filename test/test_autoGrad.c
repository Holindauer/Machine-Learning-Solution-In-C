#include "autoGrad.h"

/**
 * @helper checks if a new value with no ancestors was initialized properly
*/
void check_newValueInitialized(Value* value, int intendedValue, char opString[]){

    assert(value->ancestorArrLen == 0);
    assert(value->ancestors == NULL);
    assert(value->grad == 0);
    assert(value->value == intendedValue);
    assert(value->Backward == NULL);
    assert(strcmp(value->opString, opString) == 0);
}

/**
 * @test ensures new values are initialized properly and ancestor mechanism is working correctly
*/
void test_newValue(void){

    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    check_newValueInitialized(a, 3, "a");

    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");
    check_newValueInitialized(a, 3, "a");

    Value* c = newValue(5, NULL, NO_ANCESTORS, "c");
    check_newValueInitialized(a, 3, "a");

    Value* ancestors[3] = {a, b, c};
    int ancestorArrLen = 3;
    Value* v = newValue(9, ancestors, ancestorArrLen, "add");
    
    assert(v->value == 9);
    assert(v->grad == 0);
    assert(strcmp(v->opString, "add") == 0);

    assert(v->ancestors[0] == a);
    assert(v->ancestors[1] == b);
    assert(v->ancestors[2] == c);

    
    freeValue(&v);
    assert(v == NULL);

    freeValue(&v);
    assert(v == NULL);

    freeValue(&v);
    assert(v == NULL);

    freeValue(&v);
    assert(v == NULL);

}



int main(void){




        
}