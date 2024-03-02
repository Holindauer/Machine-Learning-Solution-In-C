#include "lib.h"


int main(void){

    /**
     * A graph stack is initialized for the purpose of releasing Value structs allocated
     * memory for during use of the program. In the context of this autograd implementation,
     * Value structs are only pushed to this stack when they are the output of computational 
     * processes. The computational graph can thus be easily deallocated by popping all nodes
     * in the stack while retaining structures like neural networks. 
    */
    GraphStack* graphStack = newGraphStack();

    printf("\nRunning Autograd Example...\n");

    // not part of the computational graph (ex: nn weights)
    Value* a = newValue(10, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(10, NULL, NO_ANCESTORS, "b");
    Value* c = newValue(10, NULL, NO_ANCESTORS, "d");

    // operations push to the computational graph
    Value* d = Add(a, b, graphStack);
    Value* e = Mul(c, d, graphStack);
    Value* f = ReLU(d, graphStack);

    // previous ancestors can be reused in the graph
    Value* g = ReLU(d, graphStack);
    Value* h = Mul(a, g, graphStack);

    /**
     * backpropagate gradient to all ancestors from singular scalar output. Backward() applies
     * backpropagation and populates the grad member of each value struct in the graph (including
     * ancestors not included in the graph stack but that have contributed to computation). 
    */
    Backward(h, NULL, NULL);

    // cleanup computational graph
    releaseGraph(graphStack);

    // cleanup initial Values
    freeValue(&a);
    freeValue(&b);
    freeValue(&c);


    printf("Done Autograd Example!\n");

    return 0;
}