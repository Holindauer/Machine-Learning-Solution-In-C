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

//-------------------------------------------------------------------------------------------------------------------------<Loss Functions>

double cross_entropy_loss(batch_outputs* outputs, int batch_size)
{
    double y_c = 0;             // initilize class/target relationship
    double example_loss = 0;    // initialize loss for a single example
    double batch_loss = 0;      // initialize loss for entire batch

    int num_classes = 10;       // number of classes in mnist
      
    for (int N = 0; N < batch_size; N++)  // N is current example of batch
    {

        example_loss = 0;                         // reset loss for the current example

        for (int c = 0; c < num_classes; c++)     // c are the potential classes a digit example could be
        {
            if ((int)outputs[N].target == c)      // check if current class of consideration is the N'th examples label
            {
                y_c = 1;         // if so, assign y_c = 1    
            }
            else                 // otherwise
            {
                y_c = 0;         // set it to 0
            }

            double prediction = predict(outputs[N].output_vector);

            // compute loss for a single example
            example_loss += y_c * log(prediction);
        }

        // add example loss to batch_loss rolling sum
        batch_loss += example_loss;
    }

    // average batch_loss across num examples
    batch_loss /= (double)batch_size;


    return batch_loss;

}