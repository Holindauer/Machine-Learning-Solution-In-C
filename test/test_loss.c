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


    // collect output of softmax into outputArr
    double* outputArr = Softmax(valueArr, vectorSize);  

     // assert that output of softmax is approximately [0.2, 0.2, 0.2, 0.2, 0.2]
    for (int i=0; i<vectorSize; i++){
        assert(fabs(outputArr[i] - 0.2) < EPSILON);
    }

    printf("PASS!\n");
}

/**
 * @test test_categoricalCrossEntropy() checks to make sure that when categorical cross entropy is applied to a 
 * Value struct ptr array, the output is correctly brought to a single Scalar and Backward() will work on it.
*/
void test_categoricalCrossEntropy(void){

    printf("test_categoricalCrossEntropy()...");


    int vectorSize = 5;

    // allocate mem for outputArr and targets arr
    Value** outputArr = malloc(sizeof(Value*) * vectorSize);
    Value** targetsArr = malloc(sizeof(Value*) * vectorSize);
    assert(outputArr != NULL);
    assert(targetsArr != NULL);

    // init valueArr and targets arr
    for (int i=0; i<vectorSize; i++){

        // values initialized to 1/i to ensure i=0 has highest magnitude for test
        outputArr[i] = newValue(exp(1), NULL, NO_ANCESTORS, "value");

        // correct class is i=0
        if (i == 0){
            targetsArr[i] = newValue(1, NULL, NO_ANCESTORS, "correct class");
        }else{
            targetsArr[i] = newValue(0, NULL, NO_ANCESTORS, "incorrect class");
        }
    }

    // get softmax results
    double* softmaxArr = Softmax(outputArr, vectorSize);

    for (int i =0; i<vectorSize; i++){
        printf("\n%lf", softmaxArr[i]);
    }

    // init graphStack for loss computation
    GraphStack* graphStack = newGraphStack();

    //  compute loss
    Value* loss = categoricalCrossEntropy(outputArr, targetsArr, softmaxArr, vectorSize, graphStack);

    printf("\nloss: %lf ", loss->value);
    
    // Backpropagate-S gradient
    Backward(loss, softmaxArr, targetsArr);

    //cleanup
    releaseGraph(graphStack);
    for (int i=0; i<vectorSize; i++){

        freeValue(&outputArr[i]);
        freeValue(&targetsArr[i]);
    }

    printf("PASS!\n");
}

int main(void){

    test_Softmax();
    test_categoricalCrossEntropy();

    return 0;
}