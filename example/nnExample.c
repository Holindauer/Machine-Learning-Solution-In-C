#include "lib.h"
#include "loadData.h"

int main(void){

    // load data
    Dataset* dataset = loadData();

    // mlp specs
    int inputSize = 4, outputSize = 3;
    int layerSizes[] = {32, 16, 8, outputSize};
    int numLayers = 4;

    // create mlp
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // training parameters
    double lr = 0.001;
    int epochs = 10;

    // loss accumulator 
    double epochLoss = 0; 

    // run training loop
    for (int epoch=0; epoch<epochs; epoch++){

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

            // accumulate loss
            epochLoss += loss->value;

            // backpropagate gradient
            Backward(loss, softmax, dataset->targets[example]);    

            // zpply gradient descent
            Step(mlp, lr);

            // zero gradient and free computational graph
            ZeroGrad(mlp);
        }

        // average epoch loss
        epochLoss /= NUM_EXAMPLES;

        printf("\nEpoch %d --- Loss: %lf", epoch, epochLoss);
    }

    // cleanup memory
    freeMLP(&mlp);
    freeDataset(&dataset);
    
    return 0;
}