#include "lib.h"
#include "loadData.h"

int main(void){

    // load data
    Dataset* dataset = loadData();

    // mlp specs
    int inputSize = 3;
    int layerSizes[] = {16, 8, 4, 1};
    int numLayers = 4;

    // create mlp
    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    // training parameters
    double lr = 0.001;
    int epochs = 10;

    // run training loop
    for (int epoch=0; epoch<epochs; epoch++){

        // forward pass on all examples
        for(int example=0; example<NUM_EXAMPLES; example++){

            // run forward pass on example
            Value** output = Forward(mlp, dataset->features[example]);

            // backpropagate gradient
            Backward(output[0]);    

            // zpply gradient descent
            Step(mlp, lr);

            // zero gradient and free computational graph
            ZeroGrad(mlp);
        }
    }

    // cleanup memory
    freeMLP(&mlp);
    freeDataset(&dataset);
    
    return 0;
}