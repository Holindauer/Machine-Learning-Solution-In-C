#include "lib.h"
#include "loadData.h"
#include "accuracy.h"

int main(void){

    // load data
    Dataset* dataset = loadData();

    // mlp specs
    int inputSize = 4, outputSize = 3;
    int layerSizes[] = {16, 8, 4, outputSize};
    int numLayers = 4;

    // create mlp
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // training parameters
    double lr = 0.001;
    int epochs = 5;

    // run training loop
    for (int epoch=0; epoch<epochs; epoch++){

        // loss accumulator 
        double epochLoss = 0, epochAccuracy = 0;

        // forward pass on all examples
        for(int example=0; example<NUM_EXAMPLES; example++){

            // run forward pass on example
            Value** output = Forward(mlp, dataset->features[example]);

            // get softmax results array
            double* softmax = Softmax(output, outputSize);

            // compute loss
            Value* loss = categoricalCrossEntropy(
                output, 
                dataset->targets[example], 
                softmax, 
                outputSize, 
                mlp->graphStack
                );
            

            // accumulate loss and accuracy
            epochLoss += loss->value;
            epochAccuracy += correctPrediction(softmax, dataset->targets[example]);

            // backpropagate gradient
            Backward(loss, softmax, dataset->targets[example]);    

            // zpply gradient descent
            Step(mlp, lr);

            // zero gradient and free computational graph
            ZeroGrad(mlp);
        }

        // average loss, acc accumulation across epoch
        epochLoss /= NUM_EXAMPLES, epochAccuracy /= NUM_EXAMPLES;

        printf("\nEpoch %d --- Loss: %lf --- Accuracy %lf", epoch, epochLoss, epochAccuracy);
    }

    // cleanup memory
    freeMLP(&mlp);
    freeDataset(&dataset);
    
    return 0;
}