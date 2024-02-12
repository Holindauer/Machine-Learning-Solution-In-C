// #include "structs.h"
// #include "macros.h"

// #define EPOCHS 15
// #define LR 0.001

// int main(void){

//     // declare 2D feature and target arrays to hold the iris data
//     // Data is represented as Value structs to allow for automatic differentiation
//     Value* features[IRIS_ROWS][IRIS_FEATURES];
//     Value* targets[IRIS_ROWS][IRIS_CLASSES];

//     // load the iris data into the feature and target arrays
//     loadData(features, targets);

//     // create a multi-layer perceptron
//     int inputSize = 4;
//     int layerSizes[] = {16, 8, 4, 1};
//     int numLayers = 4;

//     // create the multi-layer perceptron
//     MLP* mlp = createMLP(inputSize, layerSizes, numLayers); 



//     for (int example = 0; example < IRIS_ROWS; example++){

//         printf("\nExample: %d\n", example);

//         // zero the gradients of the weights and biases
//         zeroGrad(mlp);

//         printf("Gradients zeroed\n");

//         // Forward pass
//         Forward(mlp, features[example]);

//         printf("Forward pass complete\n");

//         // Backward pass
//         Backward(mlp->outputLayer->outputVector[0]);

//         printf("Backward pass complete\n");

//         // Update weights and biases
//         Step(mlp, LR);

//         printf("Weights and biases updated\n");

//         // Release the Graph
//         releaseGraph(&mlp->outputLayer->outputVector[0]);

//         printf("Graph released\n");

//     }


//     // free memory when done
//     freeDataFeatures(features, IRIS_ROWS);
//     freeDataTargets(targets, IRIS_ROWS);
//     freeMLP(mlp);

 
//     return 0;
// }


int main(void){

    return 0;
}