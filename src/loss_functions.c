#include "libraries.h"
#include "structs.h"
#include "functions.h"


/*
    This function computes the cross entropy loss for a batch during training.
    The batch struct contains both the example matrix, label array, and predictions
    array within its members.

    cross entropy for a single datapoint is defined below as:

    L(y, y_hat) = - M_sigma_c=1 (y_c * log(y_hat_c) )

    Where:  M is the number of classes

            M_sigma_c=1 is the summation starting at 1 to the num classes

            y_c is a binary value indicating whether the target label and the
            c'th label for the current example are the same.

            y_hat_c is the the model's predicted probability of whether the
            current example belongs to class c

    Thus, cross entropy for the entire batch is:

    batch_loss = -(1/N) N_sigma_i=1 ( L(y, y_hat) )

    Where N is the number of examples in the batch
*/
double cross_entropy_loss(batch_outputs* outputs, int batch_size)
{
    double y_c = 0, batch_loss = 0, prediction = 0;     

    int num_classes = 10; // num of classes in mnist
      
    for (int N = 0; N < batch_size; N++)  
    {                  

        for (int c = 0; c < num_classes; c++)            // c --- classes to compare to targets
        {
            if ((int)outputs[N].target == c) { y_c = 1;} // does the N'th example belong to class c?
            else { y_c = 0; }                          

            prediction = predict(outputs[N].output_vector);

            batch_loss += y_c * log(prediction);  // accumulate batch loss
        }
    }
    batch_loss /= (double)batch_size;             // average batch_loss across examples

    return batch_loss;
}