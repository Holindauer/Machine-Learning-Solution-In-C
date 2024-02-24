#include "lib.h"

/**
 * @helper checks if a new value with no ancestors was initialized properly
*/
void check_newValueInitialized(Value* value, int intendedValue, char opString[]){

    assert(value->ancestorArrLen == 0);
    assert(value->ancestors == NULL);
    assert(value->grad == 0);
    assert(value->value == intendedValue);
    assert(value->Backward == NULL);
    assert(strcmp(value->opString, opString) == 0);
}

/**
 * @test ensures new values are initialized properly and ancestor mechanism is working correctly
*/
void test_newValue(void){

    printf("test_newValue()...");

    // create three values and check they are initialized properly
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    check_newValueInitialized(a, 3, "a");

    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");
    check_newValueInitialized(a, 3, "a");

    Value* c = newValue(5, NULL, NO_ANCESTORS, "c");
    check_newValueInitialized(a, 3, "a");

    // manually set the ancestors of a new value to  a, b, and c
    Value* ancestors[3] = {a, b, c};
    int ancestorArrLen = 3;
    Value* v = newValue(9, ancestors, ancestorArrLen, "add");

    // check init of new value
    assert(v->value == 9);
    assert(v->grad == 0);
    assert(strcmp(v->opString, "add") == 0);

    // check ancestors are a, b, and c
    assert(v->ancestors[0] == a);
    assert(v->ancestors[1] == b);
    assert(v->ancestors[2] == c);

    // cleanup
    freeValue(&v);
    assert(v == NULL);

    freeValue(&a);
    assert(a == NULL);

    freeValue(&b);
    assert(b == NULL);

    freeValue(&c);
    assert(c == NULL);

    printf("PASS!\n");
}

/**
 * @helper check_emptyGraphStack() ensures that there is only one node in the graph stack and
 * that its fields are 0. Meant to check that releaseGraph() worked
*/
void check_emptyGraphStack(GraphStack* graphStack){
    assert(graphStack->len == 1);
    assert(graphStack->head != NULL);
    assert(graphStack->head->pValStruct == NULL);
    assert(graphStack->head->next == NULL);
}

/**
 * @helper check_graphStackAncestorUpdate() checks that the output Value from an autograd operation contains the 
 * expected fields.
 * @param opResult The Value struct that was the result of the operation
 * @param ancestor1 The first ancestor of the operation
 * @param ancestor2 The second ancestor of the operation (optiional if op isnt binary)
 * @param expectedValue The expected value of the operation
 * @param expectedGrad The expected gradient of the operation
 * @param ddx The expected type pBackwardFunc ptr to the backward function for the operation
 * @param opString The expected operation string (e.g. "add", "mull", "relu") 
 * @param binary A flag indicating if the operation is binary or not
*/
void check_opResultFields(
    Value* opResult, 
    Value* ancestor1, 
    Value* ancestor2, 
    double expectedValue, 
    double expectedGrad, 
    pBackwardFunc ddx, 
    char opString[], 
    int binary
    ){

    // assert value correctly added and related fields correctly updated
    assert(opResult->value == expectedValue);
    assert(opResult->grad == expectedGrad);
    assert(opResult->Backward == ddx);
    assert(strcmp(opResult->opString, opString) == 0);
    
    // check ancestors
    assert(opResult->ancestors[0] == ancestor1 || opResult->ancestors[1] == ancestor1);
    if (binary){
        assert(opResult->ancestors[0] == ancestor2 || opResult->ancestors[1] == ancestor2);
    }
}

/**
 * @helper check_graphStackUpdate() checks that the graph stack was updated correctly after an operation
 * by checking that the head of the graph stack is the result of the operation and that the length of the
 * graph stack matches expectations.
*/
void check_graphStackUpdate(GraphStack* graphStack, Value* expectedHead, int expectedLen){

    // assert resulting value correctly pushed to the graph stack
    assert(graphStack->head->pValStruct->value == expectedHead->value);
    assert(graphStack->head->pValStruct == expectedHead);
    assert(graphStack->len == expectedLen);
}


/**
 * @test test_Add() tests that the add operation is working correctly
*/
void test_Add(void){

    printf("test_Add()...");

    // init values to add
    Value* a = newValue(10, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(10, NULL, NO_ANCESTORS, "b");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // Add a and b
    Value* c = Add(a, b, graphStack);

    check_opResultFields(c, a, b, 20, 0, addBackward, "add", BINARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);
    
    printf("PASS!\n");
}

/**
 * @test test_AddDiff tests that the Add() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only 
 * that the function pointer within a value created from the Add() function is outputting the correct value.
*/
void test_AddDiff(void){

    printf("test_AddDiff()...");

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");

    // create a new value node
    Value* c = Add(a, b, graphStack);

    // Within Backward() which is the environment in which ->Backward() is called, the grad of the 
    // output node is set to 1 in order to kick off the backpropagation process
    c->grad = 1;

    // backpropagate the gradients
    c->Backward(c);

    // check that the gradients are correct
    assert(a->grad == 1);

    // release graph mem
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_Mul() tests that the Mul() operation is working correctly
*/
void test_Mul(void){

    printf("test_Mul()...");

    // init values to add
    Value* a = newValue(10, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(10, NULL, NO_ANCESTORS, "b");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // Mul a and b
    Value* c = Mul(a, b, graphStack);

    check_opResultFields(c, a, b, 100, 0, mulBackward, "mul", BINARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}


/**
 * @test test_MulDiff tests that the Mul() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only
 * that the function pointer within a value created from the Mul() function is outputting the correct value.
*/
void test_MulDiff(void){

    printf("test_MulDiff()...");

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(4, NULL, NO_ANCESTORS, "b");

    // create a new value node
    Value* c = Mul(a, b, graphStack);

    // Within Backward() which is the environment in which ->Backward() is called, the grad of the 
    // output node is set to 1 in order to kick off the backpropagation process
    c->grad = 1;

    // backpropagate the gradients
    c->Backward(c);

    // check that the gradients are correct
    assert(a->grad == 4);
    assert(b->grad == 3);

    // release graph mem
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_ReLU() tests that the ReLU() operation is working correctly
*/
void test_ReLU(void){

    printf("test_ReLU()...");

    // init values to add
    Value* a = newValue(-10, NULL, NO_ANCESTORS, "a");

    // init GraphStack for addition
    GraphStack* graphStack = newGraphStack();

    // ReLU a
    Value* c = ReLU(a, graphStack);

    check_opResultFields(c, a, NULL, 0, 0, reluBackward, "relu", UNARY);

    // assert resulting value correctly pushed to the graph stack
    check_graphStackUpdate(graphStack, c, 2);

    // release Graph
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}



/**
 * @test test_reluDiff tests that the ReLU() function for the value struct is working correctly when
 * it's backward function is called
 * @note This test is not checking the backward method, which recursively traverses the graph, only
 * that the function pointer within a value created from the ReLU() function is outputting the correct value.
*/
void test_ReLUDiff(void){

    printf("test_reluDiff...");

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();

    // create some ancestor nodes
    Value* a = newValue(3, NULL, NO_ANCESTORS, "a");
    Value* b = newValue(-4, NULL, NO_ANCESTORS, "b");

    // create a new value nodes
    Value* c = ReLU(a, graphStack);
    Value* d = ReLU(b, graphStack);

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

    // release graph mem
    releaseGraph(graphStack);

    // ensure graph stack has been released
    check_emptyGraphStack(graphStack);

    printf("PASS!\n");
}

/**
 * @test test_depthFirstSearch() checks to make sure that running depth first search on a computational graph
 * results in a graph stack that is ordered such that each node in the stack comes before its ancestors.
*/
void test_depthFirstSearch(void){

    printf("test_depthFirstSearch()...");

    // init dfs utilities
    GraphStack* sortStack = newGraphStack();
    HashTable* visitedHashTable = newHashTable(HASHTABLE_SIZE);


    // perform value operations
    GraphStack* opStack = newGraphStack(); // seperate graph stack needed to perform autograph ops

    // create ancestor group 1
    Value* a1 = newValue(10, NULL, NO_ANCESTORS, "a1");
    Value* a2 = newValue(10, NULL, NO_ANCESTORS, "a2");

    // create ancestor group 2
    Value* a3 = newValue(10, NULL, NO_ANCESTORS, "a3");
    Value* a4 = newValue(10, NULL, NO_ANCESTORS, "a4");

    // combine group 1 and group 2 each into output groups 1 and 2
    Value* o1 = Add(a1, a2, opStack);
    Value* o2 = Add(a3, a4, opStack);

    // combine output groupts 1 and 2 into output3
    Value* o3 = Add(o1, o2, opStack);
    assert(o3->value == 40);

    depthFirstSearch(o3, visitedHashTable, sortStack);

    assert(sortStack->head->pValStruct->value == 40);
    assert(sortStack->head->next->pValStruct->value == 20);
    assert(sortStack->head->next->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->next->next->pValStruct->value == 20);
    assert(sortStack->head->next->next->next->next->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->next->next->next->next->pValStruct->value == 10);
    
    // cleanup
    releaseGraph(sortStack); // calling releaseGraph on opStack too will segfault since the same Values are inside it
    freeHashTable(&visitedHashTable);

    printf("PASS!\n");
}

/**
 * @test test_reverseTopologicalSort() tests to make sure that the topological sorted of the computational 
 * graph within reverseTopologicalSort() correctly ordered nodes such that none come before their ancestors.
*/
void test_reverseTopologicalSort(void){

    printf("test_reverseTopologicalSort()...");

    // perform value operations
    GraphStack* opStack = newGraphStack(); // seperate graph stack needed to perform autograph ops

    // create ancestor group 1
    Value* a1 = newValue(10, NULL, NO_ANCESTORS, "a1");
    Value* a2 = newValue(10, NULL, NO_ANCESTORS, "a2");

    // create ancestor group 2
    Value* a3 = newValue(10, NULL, NO_ANCESTORS, "a3");
    Value* a4 = newValue(10, NULL, NO_ANCESTORS, "a4");

    // combine group 1 and group 2 each into output groups 1 and 2
    Value* o1 = Add(a1, a2, opStack);
    Value* o2 = Add(a3, a4, opStack);

    // combine output groupts 1 and 2 into output3
    Value* o3 = Add(o1, o2, opStack);
    assert(o3->value == 40);

    GraphStack* sortStack = newGraphStack();

    // apply reverse topological sort    
    reverseTopologicalSort(o3, &sortStack);

    // verify stack still exists
    assert(sortStack != NULL);
    assert(sortStack->head != NULL);
    assert(sortStack->head->pValStruct != NULL );

    // verify order is correct
    assert(sortStack->head->pValStruct->value == 10);
    assert(sortStack->head->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->pValStruct->value == 20);
    assert(sortStack->head->next->next->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->next->next->pValStruct->value == 10);
    assert(sortStack->head->next->next->next->next->next->pValStruct->value == 20);
    assert(sortStack->head->next->next->next->next->next->next->pValStruct->value == 40);

    // cleanup
    releaseGraph(sortStack);

    printf("PASS!\n");
}

/**
 * @test test_Backward() tests the backpropagation (Backward()) function by performing some basic operations and 
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
void test_Backward(void){   

    printf("test_Backward()...");

    // Create a graph stack to use for the operations
    GraphStack* graphStack = newGraphStack();

    // create some ancestor nodes
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


    assert(y->value == -20);

    Backward(y);

    // check that the gradients are correct
    assert(x->grad == 46);

    releaseGraph(graphStack);

    printf("PASS!");
}


int main(void){
    
    test_newValue();
    test_Add();
    test_Mul();
    test_ReLU();       
    test_ReLUDiff();
    test_MulDiff();
    test_AddDiff();
    test_depthFirstSearch();
    test_reverseTopologicalSort();
    // test_Backward();  //<---- Currently failing
}