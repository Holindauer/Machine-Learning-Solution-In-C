#include "../include/structs.h"

// test_autoGrad.c

/**
 * @test test_newValue tests the newValue function by creating a new value node and checking that it was initialized correctly
 * @dev The function creates some ancestor nodes, creates an array of ancestor nodes, and then creates a new value node.
 * @dev The function then checks that the value was initialized correctly.
*/
void test_newValue(){

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");
    Value* c = newValue(5, NULL, NO_ANCESTORS, "c");

    // create an array of ancestor nodes
    Value* ancestors[3] = {a, b, c};
    int ancestorArrLen = 3;

    // create a new value node
    Value* v = newValue(10, ancestors, ancestorArrLen, "add");

    // check that the value was initialized correctly
    assert(v->value == 10);
    assert(v->grad == 0);
    assert(v->ancestors[0] == a);
    assert(v->ancestors[1] == b);
    assert(v->ancestors[2] == c);
    assert(v->ancestors[0]->value == 3);
    assert(v->ancestors[1]->value == 4);
    assert(v->ancestors[2]->value == 5);
    assert(strcmp(v->opStr, "add") == 0);

    freeValue(v);
    freeValue(a);
    freeValue(b);
    freeValue(c);
}


/**
 * @test test_valueOperations() tests the valueOperations function by performing some basic operations and 
 * checking that the initialized correctly. This test does not check backpropagation.
*/
void test_valueOperations(void){

    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");

    // test addition
    Value* c = Add(a, b);
    assert(c->value == 7);
    assert(c->grad == 0);
    assert(c->ancestors[0] == a);
    assert(c->ancestors[1] == b);
    assert(strcmp(c->opStr, "add") == 0);

    // test multiplication
    Value* d = newValue(7, NULL, NO_ANCESTORS, "d");
    Value* e = Mul(c, d);
    assert(e->value == 49);
    assert(e->grad == 0);
    assert(e->ancestors[0] == c);
    assert(e->ancestors[1] == d);
    assert(strcmp(e->opStr, "mul") == 0);

    // test relu (positive case)
    Value* f = ReLU(e);
    assert(f->value == 49);
    assert(f->grad == 0);
    assert(f->ancestors[0] == e);
    assert(strcmp(f->opStr, "relu") == 0);    

    // test relu (negative case)
    Value* g = Mul(e, newValue(-1, NULL, NO_ANCESTORS, "neg1"));
    Value* h = ReLU(g);
    assert(h->value == 0);
    assert(h->grad == 0);
    assert(h->ancestors[0] == g);
    assert(strcmp(h->opStr, "relu") == 0);
}

/**
 * @test test_AddDiff tests that the Add() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only 
 * that the function pointer within a value created from the Add() function is outputting the correct value.
*/
void test_AddDiff(void){

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");

    // create a new value node
    Value* c = Add(a, b);

    // Within Backward() which is the environment in which ->Backward() is called, the grad of the 
    // output node is set to 1 in order to kick off the backpropagation process
    c->grad = 1;

    // backpropagate the gradients
    c->Backward(c);

    // check that the gradients are correct
    assert(a->grad == 1);

    freeValue(a);
    freeValue(b);
    freeValue(c);
}

/**
 * @test test_MulDiff tests that the Mul() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only
 * that the function pointer within a value created from the Mul() function is outputting the correct value.
*/
void test_MulDiff(void){

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");

    // create a new value node
    Value* c = Mul(a, b);

    // Within Backward() which is the environment in which ->Backward() is called, the grad of the 
    // output node is set to 1 in order to kick off the backpropagation process
    c->grad = 1;

    // backpropagate the gradients
    c->Backward(c);

    // check that the gradients are correct
    assert(a->grad == 4);
    assert(b->grad == 3);

    freeValue(a);
    freeValue(b);
    freeValue(c);
}

/**
 * @test test_reluDiff tests that the ReLU() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only
 * that the function pointer within a value created from the ReLU() function is outputting the correct value.
*/
void test_reluDiff(void){

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(-4, NULL, NO_ANCESTORS, "b");

    // create a new value nodes
    Value* c = ReLU(a);
    Value* d = ReLU(b);

    // Within Backward() which is the environment in which ->Backward() is called, the grad of the 
    // output node is set to 1 in order to kick off the backpropagation process
    c->grad = 1;
    d->grad = 1;

    // backpropagate the gradients
    c->Backward(c);
    d->Backward(d);

    // check that the gradients are correct
    assert(a->grad == 1);
    assert(b->grad == 0);

    freeValue(a);
    freeValue(b);
    freeValue(c);
    freeValue(d);
}


/**
 * @test test_Backprop() tests the backpropagation (Backward()) function by performing some basic operations and 
 * checking that the gradients are calculated correctly.
 * @dev this test case is based on the sanity check test case in karpathy's micrograd tests

Karpathy's test case:

	def test_sanity_check():

		x = Value(-4.0)
		z = 2 * x + 2 + x
		q = z.relu() + z * x
		h = (z * z).relu()
		y = h + q + q * x
		y.backward()
		xmg, ymg = x, y

		x = torch.Tensor([-4.0]).double()
		x.requires_grad = True
		z = 2 * x + 2 + x
		q = z.relu() + z * x
		h = (z * z).relu()
		y = h + q + q * x
		y.backward()
		xpt, ypt = x, y

		# forward pass went well
		assert ymg.data == ypt.data.item() #should be -20.0
		# backward pass went well
		assert xmg.grad == xpt.grad.item() # should be 46.0
*/
void test_Backprop(void){   

    // create some ancestor nodes
    Value* x = newValue(-4, NULL, NO_ANCESTORS, "x");
    Value* z = Add(
        Mul(
            newValue(2, NULL, NO_ANCESTORS, "2"), x), 
            Add(newValue(2, NULL, NO_ANCESTORS, "2"), x
            )
        );
    Value* q = Add(ReLU(z), Mul(z, x));
    Value* h = ReLU(Mul(z, z));
    Value* y = Add(Add(h, q), Mul(q, x));


    assert(y->value == -20);

    Backward(y);

    // check that the gradients are correct
    assert(x->grad == 46);

    freeValue(x);
    freeValue(z);
    freeValue(q);
    freeValue(h);
    freeValue(y);
}



/**
 * @test test_isMLPFlag tests that the isMLP flag used to differentiate between mlp related value structs and those
 * of intermediate computations is working correctly. This test checks that the mlp constructor correctly sets the
 * isMLP flag to 1 for all weights, biases, and output vectors.
 * @dev The isMLP flag by default is set to 0, and is only set to 1 under two circumstances:
 * 
 * - The first is within the construction of the mlp, weight and bias releate Value structs have their isMLP flag set to 1
 * - The second is within the Forward() call, after biases have been added, the output vector of the layer's isMLP flatg is set to 1
 * 
*/
void test_isMLPFlag(void){

    // create a multi-layer perceptron
    int inputSize = 4;
    int layerSizes[] = {16, 8, 4, 1};
    int numLayers = 4;

    // create the multi-layer perceptron
    MLP* mlp = createMLP(inputSize, layerSizes, numLayers); 

    // Checl that isMLP flag is set to 1 for the input vector
    Layer* layer = mlp->inputLayer;
    for(int i = 0; i < numLayers; i++){

        // check that outputVectors and biases are set to isMLP == 1
        for (int j = 0; j < layer->outputSize; j++){
            assert(layer->biases[j]->isMLP == 1);
            assert(layer->outputVector[j]->isMLP == 1);
        }

        // check that weights are set to isMLP == 1
        for (int j = 0; j < layer->inputSize * layer->outputSize; j++){
            assert(layer->weights[j]->isMLP == 1);
        }   

        layer = layer->next;
    }

    freeMLP(mlp);
}


/**
 * @test test_isMLPFlag tests that the isMLP flag used to differentiate between mlp related value structs and those
 * of intermediate computations is working correctly
 * @dev The isMLP flag by default is set to 0, and is only set to 1 under two circumstances:
 * 
 * - The first is within the construction of the mlp, weight and bias releate Value structs have their isMLP flag set to 1
 * - The second is within the Forward() call, after biases have been added, the output vector of the layer's isMLP flatg is set to 1
 * 
*/
void test_isMLPFlag_nonMLP(void){

    // ensure Value initialized with isMLP flag set to 0
    Value * a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value * b = newValue(4, NULL, NO_ANCESTORS, "b");
    assert(a->isMLP == 0);
    assert(b->isMLP == 0);

    // ensure addition does not set isMLP flag to 1
    Value* c = Add(a, b);
    assert(c->isMLP == 0);

    // ensure multiplication does not set isMLP flag to 1
    Value* d = Mul(a, b);
    assert(d->isMLP == 0);

    // ensure ReLU does not set isMLP flag to 1
    Value* e = ReLU(a);
    assert(e->isMLP == 0);
    
    // cleanup
    freeValue(a);
    freeValue(b);
    freeValue(c);
    freeValue(d);
    freeValue(e);   
}



/**
 * @test test_releaseGraph tests the releaseGraph function by creating a graph and then releasing it
 * @dev In the context of Backpropagation of an MLP. the releaseGraph function will recieve the output vector, which
 * in the case of this implementation is a single value struct, it must release all of the Value struct ancestors except 
 * for bias, weights, output vector. The input vector can be deallocated because we make a copy of it at the start of the 
 * forward pass.
 * @ This test makes sure that when using releaseGraph in the context of an mlp, the mlp is not deallocated in doing so.
*/
void test_releaseGraph(void){

    // create a multi-layer perceptron
    int inputSize = 4;
    int layerSizes[] = {8, 8, 4, 1};
    int numLayers = 4;

    // create the multi-layer perceptron
    MLP* mlp = createMLP(inputSize, layerSizes, numLayers); 

    // Check that isMLP flag is set to 1 for the mlp
    Layer* layer = mlp->inputLayer;
    for(int i = 0; i < numLayers; i++){

        // check that outputVectors and biases are set to isMLP == 1
        for (int j = 0; j < layer->outputSize; j++){
            assert(layer->biases[j]->isMLP == 1);
            assert(layer->outputVector[j]->isMLP == 1);
        }

        // check that weights are set to isMLP == 1
        for (int j = 0; j < layer->inputSize * layer->outputSize; j++){
            assert(layer->weights[j]->isMLP == 1);
        }   

        layer = layer->next;
    }

    // Set up an input vector
    Value** input;
    input = (Value**) malloc(inputSize * sizeof(Value*));
    for (int i = 0; i < inputSize; i++){
        input[i] = newValue(i + 8, NULL, NO_ANCESTORS, "input");

        // ensure correct val was set and that isMLP flag is set to 0
        assert(input[i]->value == i + 8);
        assert(input[i]->isMLP == 0);
    }


    // run forward pass
    Forward(mlp, input);

    // Check that isMLP flag is still set to 1 for the mlp following the forward pass
    layer = mlp->inputLayer;
    for(int i = 0; i < numLayers; i++){

        // check that outputVectors and biases are set to isMLP == 1
        for (int j = 0; j < layer->outputSize; j++){
            assert(layer->biases[j]->isMLP == 1);
            assert(layer->outputVector[j]->isMLP == 1);
        }

        // check that weights are set to isMLP == 1
        for (int j = 0; j < layer->inputSize * layer->outputSize; j++){
            assert(layer->weights[j]->isMLP == 1);
        }   

        layer = layer->next;
    }

    // backpropagate gradient of the output
    Backward(mlp->outputLayer->outputVector[0]);

    // check that the gradients are correct for the most output layer.
    assert(mlp->outputLayer->outputVector[0]->value != 0);
    assert(mlp->outputLayer->outputVector[0]->grad == 1);

    // release computation graph once gradient has been accumulated
    releaseGraph(&mlp->outputLayer->outputVector[0]);

    // ensure output node still exists
    assert(mlp->outputLayer->outputVector[0] != NULL);

    // Check that isMLP flag is still set to 1 for the mlp following the release of the graph
    layer = mlp->inputLayer;
    for(int i = 0; i < numLayers; i++){

        // check that outputVectors and biases are set to isMLP == 1
        for (int j = 0; j < layer->outputSize; j++){

            // ensure isMLP flag is set to 1
            assert(layer->biases[j]->isMLP == 1);
            assert(layer->outputVector[j]->isMLP == 1);

            // ensure op strings still in tact
            assert(strcmp(layer->biases[j]->opStr, "initBiases") == 0); // <-- should lways be initBiases
            assert( // output vector opStr is expected to be overwritten. Can be any of these
                strcmp(layer->outputVector[j]->opStr, "initOutputVector") == 0 ||
                strcmp(layer->outputVector[j]->opStr, "add") == 0||
                strcmp(layer->outputVector[j]->opStr, "mul") == 0 ||
                strcmp(layer->outputVector[j]->opStr, "relu") == 0
            );
        }

        // check that weights are set to isMLP == 1
        for (int j = 0; j < layer->inputSize * layer->outputSize; j++){
            // ensure isMLP flag is set to 1
            assert(layer->weights[j]->isMLP == 1);

            // ensure op strings still in tact
            assert(strcmp(layer->weights[j]->opStr, "initWeights") == 0); // <-- should lways be initWeights
        }   

        layer = layer->next;
    }



    // cleanup
    freeMLP(mlp);
    for (int i = 0; i < inputSize; i++){
        freeValue(input[i]);
    }
    free(input);
}


// run the tests
int main(void){

    printf("Running Autograd tests...\n");

    test_newValue();
    test_valueOperations();
    test_AddDiff();
    test_MulDiff();
    test_reluDiff();
    test_Backprop(); 
    test_releaseGraph();
    test_isMLPFlag();
    test_isMLPFlag_nonMLP();

    printf("All Autograd Tests Passed!\n\n");

    return 0;
}