#include "lib.h"


/**
 * @note this function applies softmax activation to an array of value pointers. 
 * @dev softmax is a vector of the exponential of each value in the input vector divided by the sum of all exponentials of 
 * input vector elements.
 * @dev Softmax() is not being considered an autograd operation by itself, instead it is considered a helper function for 
 * categoricalCrossEntropy(). For this reason there is no direct softmaxBackward() function for computing partial derivatives 
 * for Softmax().
 * @param valueArr an array of Value struct pointers to apply softmax to
 * @param lenArr length of each array
 * @return a dynamically allocated array of the softmax outputs
*/
double* Softmax(Value** valueArr, int lenArr){

    // allocate memory for sotmax array
    double* softmax = malloc(sizeof(double) * lenArr);
    assert(softmax != NULL);

    // softmax numerator
    double expSum = 0;
    for (int class=0; class<lenArr; class++){
        expSum += exp(valueArr[class]->value);
    }  

    //  compute softmax
    for(int class=0; class<lenArr; class++){
        softmax[class] = exp(valueArr[class]->value) / (expSum + EPSILON);
    }

    return softmax;
}

/**
 * freeSoftmax() frees the array of doubles created by the Softmax() function
 * @param softmaxArr ptr to array of doubles
*/
void freeSoftmax(double** softmaxArr){
    assert(softmaxArr != NULL);
    free(*softmaxArr);  
    *softmaxArr = NULL;
}

/**
 * @note categoricalCrossEntropyBackward() computes the gradient of the single value output from categoricalCrossEntropy()
 * wrt to its immediate ancestors. 
 * @note that this is the gradient of the composite function of categoricalCrossEntropy( Softmax() )
 * @dev the partial derivatives of the compos
 * @param v a Value struct ptr that is the output of categoricalCrossEntropy() 
 * @param softmaxOutput an array of doubles containing the outputs to softmax(mlp output)
 * @param targetsArr traget vector array
*/
void categoricalCrossEntropyBackward(Value* v, double* softmaxOutput, Value** targetsArr, int lenArr){
    assert(v!= NULL);
    assert(v->ancestors != NULL);

    // Propagate the gradient to all ancestors
    for(int i = 0; i<lenArr; i++){
        if (v->ancestors[i] != NULL){

            // apply chain rule
            v->ancestors[i]->grad += v->grad * (softmaxOutput[i] - targetsArr[i]->value);
        }
    }
}


/**
 * The current issue is that the backward function for categorical cross entropy requires more information 
 * that just the immediate ancestors and their scalar vallues. It requires as well the predicted probabilities
 * after applying sotmax and the labes for each class in the target vector. The reason this is an issue is 
 * because the pBackwardFunc function pointer that was set up for the eventual backpropagation of the gradient
 * only accepts a single input and thus cannoth supply that information into the function when called. 
 * 
 * One potential solution to this would be to include another function pointer within the Value struct if which
 * the correct function pointer to use could be determined within the Backward() function. 
 * 
 * This would require:
 *  - modifying Backward() to include that logic (adjusting the function header)
 *  - Adjusting the way that Softmax() and categoricalCrossEntropy() are called in conjunction to each other,
 *    which will probably mean Softmax() is called in whatever main function the training loop is called within
*/


/**
 * @note categoricalCrossEntropy() applies the softmax activation functions to an input vector (Value* array) and then 
 * applies the categorical cross entropy loss function to bring the vector input to a scalar output that can be passed
 * into the Backward() function for backpropagation of the broader graph.
 * @dev cross entropy is the negative summation of the log of the predicted probabilites that each example is its true 
 * class, multiplied by the target label for that class (each element of the one hot vector)
 * @dev the final ReLU() activation in the output of an mlp will protect against a negative input to the log computation
 * @dev softmax is applied here as opposed to within the mlp to reduce the complexity of the derivative computation by
 * taking advantage of the simplification of each partial derivative of the gradient of softmax as the inner function of 
 * a composition with categorical cross entropy. These partial derivatives end up being the predicted probability of
 * each class subtracked from the whether that target is correct.
 * @param outputArr is the output vector array of an MLP struct that is the same length as the number of classes
 * @param targetsArr is the array of targets (ie, the one hot encoded label for the current example).
 * @param lenArr is the length of both valueArr and targetsArr
 * @param graphStack is the graph stack of the mlp of which the valueArr output came from
*/
Value* categoricalCrossEntropy(
    Value** outputArr, 
    Value** targetsArr, 
    double* softmaxOutput, 
    int lenArr, 
    GraphStack* graphStack
    ){

    // compute value of loss
    double lossSum = 0;
    for (int class=0; class<lenArr; class++){

        // accumulate negative log(probability) * class label
        lossSum -= log(outputArr[class]->value) * targetsArr[class]->value;
    }   

    // place loss into Value struct. Note that the valueArr being passed into newValue() 
    // will have a deep copy of itself constructed for the new loss Value. 
    Value* loss = newValue(lossSum, outputArr, lenArr, "loss");

    // push to the graph stack
    pushGraphStack(graphStack, loss);

    // set backward function ptr 
    loss->BackwardLoss = categoricalCrossEntropyBackward;

    return loss;
}