#include "lib.h"


/**
 * @test test_Softmax() tests that the setup for the Softmax() helper function in loss.c computes an expected result.
*/
void test_Softmax(void){

    printf("test_Softmax()...");

    int vectorSize = 5;

    // init valueArr to values of e
    Value** valueArr = malloc(sizeof(Value*) * vectorSize);
    for (int i=0; i<vectorSize; i++){
        valueArr[i] = newValue(exp(1), NULL, NO_ANCESTORS, "value");
    }

    // init outputArr
    double outputArr[5] = {0};

    // collect output of softmax into outputArr
    Softmax(valueArr, outputArr, vectorSize);  

     // assert that output of softmax is approximately [0.2, 0.2, 0.2, 0.2, 0.2]
    for (int i=0; i<vectorSize; i++){
        assert(fabs(outputArr[i] - 0.2) < EPSILON);
    }

    printf("PASS!\n");
}


int main(void){

    test_Softmax();

    return 0;
}