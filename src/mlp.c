#include "libraries.h"
#include "macros.h"
#include "structs.h"


/**
 * @notice mlp.c contains the implementation of a constructor for a multi-layer perceptron
 * @dev Each layer of an mlp is a node in a doubly linked list
 * @dev Each layer contains a weight matrix and a bias vector. These parameters are represented as
 * Value structs in order to support backpropagation using the autograd implementation from autoGrad.c
*/


/**
 * @notice initWeights() is a helper function for the createMLP() constructor. It initializes the weights
 * (Value structs) for a given layer with random values between -1 and 1. 
*/
Value** initWeights(int inputSize, int outputSize){

    Value** weights = (Value**)malloc(inputSize * outputSize * sizeof(Value*));

    for(int i = 0; i < (inputSize * outputSize); i++){

        // create random value between -1 and 1
        float randomFloat = (float)rand() / (RAND_MAX + 1u) * 2.0f - 1.0f;
        weights[i] = newValue(randomFloat, NULL, NO_ANCESTORS, "initWeights");
    }

    return weights;
}

/**
 * @notice initBiases() is a helper function for the createMLP() constructor. It initializes the biases
 * (Value structs) for a given layer with random values between -1 and 1. 
*/
Value** initBiases(int outputSize){

    Value** biases = (Value**)malloc(outputSize * sizeof(Value*));

    for(int i = 0; i < outputSize; i++){

        // create random value between -1 and 1
        float randomFloat = (float)rand() / (RAND_MAX + 1u) * 2.0f - 1.0f;
        biases[i] = newValue(randomFloat, NULL, 0, "initBiases");
    }

    return biases;
}

/**
 * @notice initOutputVector() is a helper function for the forward pass of the network. It initializes the output
 * vector for a given layer with Value structs that have a value of 0 and no ancestors.
*/
Value** initOutputVector(int outputSize){

    Value** output = (Value**)malloc(outputSize * sizeof(Value*));
    assert(output != NULL);

    for(int i = 0; i < outputSize; i++){
        output[i] = newValue(0, NULL, NO_ANCESTORS, "initOutputVector");
    }

    return output;
}

/**
 * @notice createMLP() is a constructor function that creates a multi-layer perceptron with the specified
 * layer sizes. The weights and biases are initialized with random values between -1 and 1. 
 * @dev The MLP struct is used to store the Layer structs in a doubly linked list. The head and tail pointers
 * are used to access the start and end of the network.
 * @param inputSize The length of the input feature vector
 * @param layerSizes An array of integers representing the number of neurons in each layer of the network
*/
MLP* createMLP(int inputSize, int layerSizes[], int numLayers){

    // allocate memory for the MLP struct
    MLP* mlp = (MLP*)malloc(sizeof(MLP));  
    assert(mlp != NULL);

    // allocate memory for the graph stack
    mlp->graphStack = newGraphStack();

    // allocate memory for the input layer
    Layer* inputLayer = (Layer*)malloc(sizeof(Layer));
    assert(inputLayer != NULL);

    // set input layer fields
    inputLayer->inputSize = inputSize;  
    inputLayer->outputSize = layerSizes[0];
    inputLayer->prev = NULL;

    // initialize weights and biases for input layer
    inputLayer->weights = initWeights(inputSize, layerSizes[0]);
    inputLayer->biases = initBiases(layerSizes[0]);
    inputLayer->outputVector = initOutputVector(layerSizes[0]);

    // check that memory was allocated
    assert(inputLayer->weights != NULL);
    assert(inputLayer->biases != NULL);
    assert(inputLayer->outputVector != NULL);

    // set links for the input layer of mlp
    mlp->inputLayer = inputLayer;
    mlp->numLayers = numLayers;

    // allocate create the rest of the layers
    for (int i=1; i<numLayers; i++){

        // allocate memory for the current layer
        Layer* currentLayer = (Layer*)malloc(sizeof(Layer));
        assert(currentLayer != NULL);

        // set current layer fields
        currentLayer->inputSize = layerSizes[i-1];
        currentLayer->outputSize = layerSizes[i];
        currentLayer->prev = inputLayer;

        // initialize weights and biases for current layer
        currentLayer->weights = initWeights(layerSizes[i-1], layerSizes[i]); 
        currentLayer->biases = initBiases(layerSizes[i]);
        currentLayer->outputVector = initOutputVector(layerSizes[i]);

        // check that memory was allocated
        assert(currentLayer->weights != NULL);
        assert(currentLayer->biases != NULL);
        assert(currentLayer->outputVector != NULL);

        // set next pointer for input layer
        inputLayer->next = currentLayer;

        // set links for the most recent layer
        currentLayer->next = NULL;
        currentLayer->prev = inputLayer;

        // update input layer pointer to current layer
        inputLayer = currentLayer;
    }

    // set output layer pointer
    mlp->outputLayer = inputLayer;

    return mlp;
}


/**
 * @notice zeroGrad() is used in relation to the backward pass of the network. It resets the gradient of all weights
 * and biases to 0.
 * @dev In addition to zeroing the gradients, zeroGrad() also handles deallocation of the computational graph collected 
 * during the forward pass.  
 * @dec As well, in order to ensure that the computational graph for the next example/epoch is new, the MLP argument is
 * copied and dealocated
 * @param mlp is a pointer to a pointer to an MLP struct to zero the gradients of.
 * @param layerSizes[] is an array of the output sizes of the mlp
 * @param intputSize is input vector size for the mlp
 * @param numLayers is num layers of the mlp
*/
void zeroGrad(MLP** mlp, int inputSize, int layerSizes[], int numLayers){

    // create a new, blank mlp by calling createMLP
    MLP* newMLP = createMLP(inputSize, layerSizes, numLayers); 
    assert(newMLP != NULL);
    assert(newMLP->numLayers == (*mlp)->numLayers);

    // grab the first layers of both mlps
    Layer* ogLayer = (*mlp)->inputLayer;
    Layer* newLayer = newMLP->inputLayer;
    
    // copy the og mlp weights, biases, other specs into the new mlp
    // @dev this should zero the gradients by their initialization so there is no need to zero them again
    for (int i = 0; i < (*mlp)->numLayers; i++){

        // ensure the new mlp has the same dimmensions as the old mlp
        assert(ogLayer->inputSize == newLayer->inputSize);
        assert(ogLayer->outputSize == newLayer->outputSize);

        // copy weights
        for(int j = 0; j < ogLayer->inputSize * ogLayer->outputSize; j++){

            // copy the value of the weight
            newLayer->weights[j]->value = ogLayer->weights[j]->value;
        }
        for(int j = 0; j < ogLayer->outputSize; j++){
            
            // copy the value of the bias
            newLayer->biases[j]->value = ogLayer->biases[j]->value;
        }

        // move to the next layer in both mlps
        ogLayer = ogLayer->next;
        newLayer = newLayer->next;
    }

    // release the graph from and free old mlp
    releaseGraph((*mlp)->graphStack);
    freeMLP(*mlp);

    // set the pointer to the new mlp
    *mlp = newMLP;
}


/**
 * @notice freeWeights() is a helper function for the createMLP() destructor. It frees the memory allocated
 * for the weights of a given layer.
*/
void freeWeights(Value** weights, int inputSize, int outputSize){
    for(int i =0; i < (inputSize * outputSize); i++){
        freeValue(weights[i]);
    }
    free(weights);
}

/**
 * @notice freeBiases() is a helper function for the createMLP() destructor. It frees the memory allocated
 * for the biases of a given layer.
*/
void freeBiases(Value** biases, int outputSize){
    for(int i =0; i < outputSize; i++){
        freeValue(biases[i]);
    }
    free(biases);
}

/**
 * @notice freeMLP() is a destructor function that frees the memory allocated for the MLP struct and all of its
 * associated layers, weights, and biases.
*/
void freeMLP(MLP* mlp){

    Layer* layer =mlp->inputLayer;
    Layer* nextLayer = layer; 

    for (int i = 0; i < mlp->numLayers; i++){

        // move to the next layer
        layer = nextLayer;

        // free the weights, biases, and output vector
        for(int j = 0; j < layer->inputSize * layer->outputSize; j++){
            freeValue(layer->weights[j]);
        }
        for(int j = 0; j < layer->outputSize; j++){
            freeValue(layer->biases[j]);
        }
        free(layer->weights);
        free(layer->biases);
        free(layer->outputVector);
        
        // set next layer then free current layer
        nextLayer = layer->next;
        free(layer);
   }

   free(mlp);
}

