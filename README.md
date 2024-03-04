# Autodiff and Neural Net Implementation in C

This repository contains an implementation of reverse mode automatic differentiation for the training of simple multi layer perceptron neural networks for the task of classification. This entire implementation was written purely in less than 600 lines of C. 

Also, Thanks to Andrej Karpathy for the inspo for this project. This C implementation is based on [micrograd](https://github.com/karpathy/micrograd) 

# What is Autodiff?

Reverse mode automatic differentiation (Autodiff) is a technique for computing the gradient (vector of partial derivatives) of a parametric function that maps from $\mathbb{R^n} \rightarrow \mathbb{R}$. In the context of deep learning, computing the gradient of the loss of a neural network is required for the gradient descent algorithm, which is the algorithm that adjusts the weights of a neural network towards better performance.

Autodiff is a protocol that programatically computes the gradient of functions too complicated to find an explicit derivative expression of. Autodiff gets around this issue by implementing backpropagation, the process of computing the gradient of a function by recursively applying the calculus chain rule backwards from the output state. Backprop is done programatically in autodiff by tracking a directed acyclic graph of all state during the computation that went into producing some single scalar value. By splitting up a large function into its elementary components (adds, multiplies, ReLUs...), of which their partial derivatives are very simple, each partial derivative of an extremely complicated function is constructed backwards from chaining together these simple components' derivatives wrt their imediate ancestors. This process starts with at the final output and moves backwards until all parameters are reached.

This repository implements this algorithm, and uses it to train a multi layer perceptron neural network of variable size at runtime. 

# How is Autodiff Represented in this Repository?

The autodiff implementation is built around the Value struct. A Value has the following 4 important members (there are more members than this in the code, but I have included only the most important ones here for understanding autodiff):

    typedef struct _value {
        double value;             
        double grad;              
        pBackwardFunc Backward; 
        Value** ancestors;
    } Value;

A Value contains a *value*, this is an intermediate state at some point in the computational graph (A multiplication within a dot product for example). It also has the *grad* member, this represents the partial derivative of the elementary operation that went into creating this particular Value's *value*. The *ancestors* member is an array of Value struct pointers containing the immediate ancestors that went into creating this particular Value. Finally, *pBackwardFunc* is a function pointer that will compute the partial derivative of the particular elementary operation that created this Value wrt to its ancestors. 

Ancestors structs are linked together in a directed acyclic graph as computation occurs. For example, if I create two Values, and add them together, 

    Value* A = newValue(5.3, NULL, NO_ANCESTORS, "a"); // Value of 5.3 w/ no ancestors and label "a"
    Value* B = newValue(10, NULL, NO_ANCESTORS, "b");

    Value* C = Add(a, b, graphStack); // graphStack collects Values for deallocation

What is happening here under the hood is that a new Value struct, C, is created with Values A and B linked to it in the *ancestors* array. As well as the derivative function for Add() is being set within C's pBackwardFunc pointer for backpropagation. The operations that are currently implemented are: 

    - Add()   
    - Mul()   
    - ReLU()   
    - SoftMax()  // Sort of, softmax is computed w/ cross entropy. See loss.c for expl
    - categoricalCrossEntropy()  

A long chain of computation can be built up using these operations. And the gradient of the resulting function (the particular string of operations) is computed recursively by calling Backward() w/ the final output.

    GraphStack* graphStack = newGraphStack();

    // Do Computation
    Value* x = newValue(-4, NULL, NO_ANCESTORS, "x");

    Value* z = Add(
        Mul(
            newValue(2, NULL, NO_ANCESTORS, "2"), x, graphStack), 
            Add(newValue(2, NULL, NO_ANCESTORS, "2"), x, graphStack),
            graphStack
        );
    
    Value* q = Add(ReLU(z, graphStack), Mul(z, x, graphStack), graphStack);
    Value* h = ReLU(Mul(z, z, graphStack), graphStack);
    Value* y = Add(Add(h, q, graphStack), Mul(q, x, graphStack), graphStack);

    // Backpropagate Gradient
    Backward(y, NULL, NULL);  

    assert(y->value == -20);    
    assert(x->grad == 46);

    // free memory in graph
    releaseGraph(&graphStack);


The code above is a test from Andrej Karpathy's autograd implementation in python, [micrograd](https://github.com/karpathy/micrograd). 

# How are Neural Networks Represented in this Repository?

This repository implements multi layer perceptron neural networks for the task of classification. MLPs can be created with the newMLP() function, which accepts network specification arguments.

    int inputSize = 4;
    int layerSizes[] = {32, 16, 8, 4};
    int numLayers = 4;

    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

Under the hood, the mlp is a dynamically allocated linked list of Layer structs, each containg two arrays of Value struct ptrs for weight matrices and bias vectors. Weights and biases within layers are a seperate system than the computation graph, and will not be deallocated by calling releaseGraph() on the final ouput of an mlp. 


    Value** output = Forward(mlp, example);  // example of type Value**

# Simple MLP Training

MLP training can be done in relatively few lines of code. A major goal of this project was to make the syntax for mlp training as close to that of PyTorch as possible. Here is the simplest training loop you can construct using this repository, this is a simplified version of the example in example/nnExample.c:

    Dataset* dataset = loadData();

    int inputSize = 4, outputSize = 3;
    int layerSizes[] = {16, 8, 4, outputSize};
    int numLayers = 4;

    MLP* mlp = newMLP(inputSize, layerSizes, numLayers);

    double lr = 0.001;
    int epochs = 10;

    // training loop
    for (int epoch=0; epoch<epochs; epoch++){

        // iterate examples
        for(int example=0; example<NUM_EXAMPLES; example++){

            // forward pass 
            Value** output = Forward(mlp, dataset->features[example]);

            // apply softmax 
            double* softmax = Softmax(output, outputSize);

            // compute loss
            Value* loss = categoricalCrossEntropy(
                output, dataset->targets[example], softmax, outputSize, mlp->graphStack
                );
            
            // backpropagate gradient
            Backward(loss, softmax, dataset->targets[example]);    

            // apply gradient descent
            Step(mlp, lr);

            // zero gradient and free computational graph
            ZeroGrad(mlp);
        }
    }

    // cleanup 
    freeMLP(&mlp);
    freeDataset(&dataset);

Note: mlp training is bit fragile. Currently, the example in example/nnExample.c shows much improvement across epoch steps but little across epochs. This doesn't appear to be an issue with autograd, potentially with softmax/crossEntropy, or just limited deep learning techniques implemented.

# Extra Thoughts

This project was a fairly intense software engineering challenge, in addition to a great consolodation of my understanding of backpropagation. I was fairly happy with how concise the code ended up being, as simple code was a consistent goal throughout development.

    -------------------------------------------------------------------------------
    Language                     files          blank        comment           code
    -------------------------------------------------------------------------------
    C                                7            266            424            469
    C/C++ Header                    10             49             90            126
    -------------------------------------------------------------------------------
    SUM:                            17            315            514            595
    -------------------------------------------------------------------------------