#include "lib.h"
#include "loadData.h"

/**
 * @note correctPrediction() determines whether the highest probability within the output of softmax 
 * is accurate to the one hot encoded target vector array of Value struct ptrs
*/
double correctPrediction(double* softmaxOutputs, Value** target){

    int indexTargetClass = -1;
    int indexHighestProbability = -2;

    // perform an argmax on both arrays
    for(int idx = 0; idx<NUM_CLASSES; idx++){
        
        if (target[idx]->value == 1){
            indexTargetClass = idx;
        }

        if (softmaxOutputs[idx] > indexHighestProbability){
            indexHighestProbability = idx;
        }
    }

    // compare argmaxes
    if (indexTargetClass == indexHighestProbability){
        return 1;
    }

    return 0;
}