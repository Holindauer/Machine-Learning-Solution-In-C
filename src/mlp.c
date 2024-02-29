#include "lib.h"

// ---------------------------------------------------------------------------------------------------------------------- MLP Constructors

/**
 * @note randDouble is a helper function that returns a double value between -1 and 1
*/
double randDouble(void){
    return (double)rand() / (RAND_MAX + 1u) * 2.0f - 1.0f;
}

/**
 * @note newLayer allocates memory for and intializes a new Layer struct. 
 * @dev weights and biases within the Value struct ptr arrays are initialized to random doubles between -1 and 1
 * @dev output vector array of Value struct ptrs initialized to Value structs of zeroes
 * @param inputSize
 * @param outputSize
*/
Layer* newLayer(int inputSize, int outputSize){

    // allocate mem for layer
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    assert(layer != NULL); 

    // set layer dimmensions
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;

    // init layer links to NULL
    layer->next = NULL, layer->prev = NULL;

    // allocate mem for weights, biases, hidden/output state
    layer->weights = (Value**)malloc(inputSize * outputSize * sizeof(Value*));
    layer->biases = (Value**)malloc(outputSize * sizeof(Value*));
    assert(layer->weights != NULL && layer->biases != NULL);

    // init weights
    for (int i = 0; i < (inputSize * outputSize); i++){

        // weights initialized between -1 and 1
        layer->weights[i] = newValue(randDouble(), NULL, NO_ANCESTORS, "init weights");
        assert(layer->weights[i] != NULL);
    }

    // init biases and output/hidden state
    for (int i = 0; i < outputSize; i++){

        // init biases between -1 and 1
        layer->biases[i] = newValue(randDouble(), NULL, NO_ANCESTORS, "init biases");
        assert(layer->biases[i] != NULL);
    }

    return layer;
}


/**
 * @note newMLP() is a constructor for an MLP struct containing a listed list of Layer structs 
 * @param inputSize the length of the input feature vector
 * @param layerSizes An array of integers representing the number of neurons in each layer of the network
 * @param numLayers
*/
MLP* newMLP(int inputSize, int layerSizes[], int numLayers){

    // allocate mem for layer
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    assert(mlp != NULL);

    // create graph stack
    mlp->graphStack = newGraphStack();
    assert(mlp->graphStack != NULL);
    
    // create input layer
    Layer* prevLayer = newLayer(inputSize, layerSizes[0]); 
    assert(prevLayer != NULL);

    // set link to input layer
    mlp->inputLayer = prevLayer;

    // temp for iteration
    Layer* currentLayer = NULL;

    // create the rest of the layers
    for (int i=1; i<numLayers; i++){

        // allocate mem and init layer
        currentLayer = newLayer(layerSizes[i-1], layerSizes[i]);
        assert(currentLayer != NULL);

        // set links
        prevLayer->next = currentLayer;
        currentLayer->next = NULL;
        currentLayer->prev = prevLayer;

        // update prev layer
        prevLayer = currentLayer;
    }

    // set link to output layer
    mlp->outputLayer = prevLayer;

    return mlp;
}

// ---------------------------------------------------------------------------------------------------------------------- MLP Destructors

/**
 * @note freeLayer() frees a layer struct and all memory witin it. This includes all Value structs in the weights and biases
 * @param layer ptr to Layer struct ptr
*/
void freeLayer(Layer** layer){

    // free Value structs in weights
    for (int i = 0; i < (*layer)->outputSize * (*layer)->inputSize; i++){

        freeValue(&((*layer)->weights[i]));
        assert((*layer)->weights[i] == NULL);
    } 

    // free Value structs in biases and output vector
    for (int i = 0; i < (*layer)->outputSize; i++){

        // free bias
        freeValue(&((*layer)->biases[i]));
        assert((*layer)->biases[i] == NULL);    
    }

    // free weights, bias, output arrays of Value struct ptrs
    free((*layer)->weights);
    free((*layer)->biases);

    (*layer)->weights = NULL;
    (*layer)->biases = NULL;

    // free layer struct
    free(*layer);
    *layer = NULL;
}

/**
 * @note free mlp frees all memory inside an mlp struct
 * @param mlp ptr to MLP struct ptr
*/
void freeMLP(MLP** mlp){

    // get first layer
    Layer* layer = (*mlp)->inputLayer;
    Layer* next = NULL;

    // free all layers
    while (layer != NULL){

        next = layer->next;

        freeLayer(&layer);
        assert(layer == NULL);

        layer = next;
    }

    // release graph stack
    releaseGraph((*mlp)->graphStack);

    // free mlp struct
    free(*mlp);
    *mlp = NULL;
}

