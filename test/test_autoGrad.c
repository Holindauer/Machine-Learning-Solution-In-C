#include "libraries.h"
#include "../src/autoGrad.c"
#include "../include/macros.h"


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
 * @test test_backpropagation() tests the backpropagation function by performing some basic operations and 
 * checking that the gradients are calculated correctly.
 * @dev this test case is based on the sanity check test case in karpathy's micrograd tests
*/

/**
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
        Mul(newValue(2, NULL, NO_ANCESTORS, "2"), x), Add(newValue(2, NULL, NO_ANCESTORS, "2"), x)
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