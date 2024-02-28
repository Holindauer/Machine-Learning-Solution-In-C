#include "lib.h"


/**
 * @note newOutputVector initializes an array of Value ptrs with values set to zero 
 * @note output vectors can be freed within the forward pass, the are not needed beyond the forward pass (except
 * the final output vector). However, if their Value ptrs are freed within the forward pass, Backward() will not
 * work. Value ptrs in output vectors should be deallocated with releaseGraph()
 * @param outputSize
 * @returns array of value struct ptrs
*/
Value** newOutputVector(int outputSize){

    // allocate mem
    Value** output = (Value**)malloc(sizeof(Value*) * outputSize);
    assert(output != NULL);

    // init values to 0 for dot product accumulatiaon
    for (int i=0; i<outputSize; i++){
        output[i] = newValue(0, NULL, NO_ANCESTORS, "init output");
        assert(output[i] != NULL);
    }

    return output;
}

/**
 * @note MultiplyWeights() computes matrixt vector multiplication of a Layer struct's weight matrix with an 
 * input vector using Value operations from autograd.c
 * @param layer the layer in which to use its weight matrix
 * @param input input vecctor represented as array of Value struct ptrs 
 * @param graphStack the graph stack of the mlp of which the layer came from
 * @return output vector represented as array of  Value struct ptrs
*/
Value** MultiplyWeights(Layer* layer, Value** input, GraphStack* graphStack){

    // init output vector 
    Value** output = newOutputVector(layer->outputSize); 
    // ! memory leak here from initial Value structs inside output vector created from newOutputVector(). not super 
    // ! important at the moment, but worth noting that those Value's will not be part of the computational graph 

    // iterate over each output neuron
    for (int i=0; i<layer->outputSize; i++){

        // Accumulate dot product of i'th row weights w/ input vector
        for (int j=0; j < layer->inputSize; j++){

            output[i] = Add(
                output[i],
                Mul(layer->weights[i * layer->inputSize + j], input[j], graphStack),
                graphStack
            );
        }
    }

    return output;
}

/**
 * @note AddBias performs elementwise vector addition on the output of MultiplyWeights() with the biases from 
 * a Layer struct using the Value operations from autograd.c
 * @dev Elementwise addition is done in place on the input array
 * @param layer the layer in which to use its bias matrix
 * @param input input vecctor represented as array of Value struct ptrs 
 * @param graphStack the graph stack of the mlp of which the layer came from
 * @return A ptr to the input vector array
*/
Value** AddBias(Layer* layer, Value** input, GraphStack* graphStack){

    // Add biases to input vector in place
    for (int i = 0; i<layer->outputSize; i++){
        input[i] = Add(layer->biases[i], input[i], graphStack);
    }

    // retrun ptr to input vector acted on in place
    return input;
}

/**
 * @note ApplyReLU() applies elementwise ReLU() to the input vector array of Value Struct ptrs
 * @dev Elementwise ReLU is done in place on the input array
 * @param layer the layer of which this output vector is being computed for (to get dimmensions)
 * @param input input vecctor represented as array of Value struct ptrs 
 * @param graphStack the graph stack of the mlp of which the layer came from
 * @return A ptr to the input vector array
*/
Value** ApplyReLU(Layer* layer, Value** input, GraphStack* graphStack){

    // applie relue to all elements of the input
    for (int i = 0; i<layer->outputSize; i++){
        input[i] = ReLU(input[i], graphStack);
    }

    // return ptr to input vector acted on in place
    return input;
}


/**
 * @note Forward() is used to perform the forward pass of an MLP struct. 
 * @dev 
 * 
 * @returns an array of Value struct pointers representing the final output of the network
*/
Value** Forward(MLP* mlp, Value** input){

    // retrieve input layer
    Layer* layer = mlp->inputLayer;

    // compute output of first layer 
    Value** output = MultiplyWeights(layer, input, mlp->graphStack);
    output = AddBias(layer, output, mlp->graphStack);
    output = ApplyReLU(layer, output, mlp->graphStack);

    // move to next layer before starting loop
    layer = layer->next;

    // compute next hidden states
    while(layer != NULL) {

        // compute layer output
        output = MultiplyWeights(layer, output, mlp->graphStack);
        output = AddBias(layer, output, mlp->graphStack);
        output = ApplyReLU(layer, output, mlp->graphStack);
    
        // move up one layer
        layer = layer->next;
    }

    return output;
}