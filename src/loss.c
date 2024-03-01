#include "lib.h"


/**
 * @note this function applies softmax activation to an array of value pointers. 
 * @dev softmax is a vector of the exponential of each value in the input vector divided by the sum of all exponentials of 
 * input vector elements.
 * @dev Softmax() is not being considered an autograd operation by itself, instead it is considered a helper function for 
 * categoricalCrossEntropy(). For this reason there is no direct softmaxBackward() function for computing partial derivatives 
 * for Softmax().
 * @param valueArr an array of Value struct pointers to apply softmax to
 * @param outputArr an array of doubles for containing the outputs of softmax. must be initialized to zeros.
 * @param lenArr length of each array
*/
void Softmax(Value** valueArr, double outputArr[], int lenArr){

    // softmax numerator
    double expSum = 0;
    for (int class=0; class<lenArr; class++){
        expSum += exp(valueArr[class]->value);
    }  

    //  compute softmax
    for(int class=0; class<lenArr; class++){
        outputArr[class] = exp(valueArr[class]->value) / (expSum + EPSILON);
    }
}


void categoricalCrossEntropyBackward(){
}

/**
 * @note categoricalCrossEntropy() applies the softmax activation functions to an input vector (Value* array) and then 
 * applies the categorical cross entropy loss function to bring the vector input to a scalar output that can be passed
 * into the Backward() function for backpropagation of the broader graph.
 * @dev cross entropy is the negative summation of the log of the predicted probabilites of that the example is in each
 * class multiplied by the target label for that class (each element of the one hot vector)
 * @dev the final ReLU() activation in the output of an mlp will protect against a negative input to the log computation
 * @dev softmax is applied here as opposed to within the mlp to reduce the complexity of the derivative computation by
 * taking advantage of the simplification of each partial derivative in the gradient of when categorical cross entropy 
 * and softmax are composed as a composite function. These partial derivatives end up being the predicted probability of
 * each class subtracked from the whether that target is correct.
 * @param valueArr is the output vector array of an MLP struct that is the same length as the number of classes
 * @param targetsArr is the array of targets (ie, the one hot encoded label for the current example).
*/
Value* categoricalCrossEntropy(Value** valueArr, Value** targetsArr){




}