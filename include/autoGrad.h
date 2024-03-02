#pragma once
#include "lib.h"
#include "graphStack.h"
#include "mlp.h"

// autoGrad.h


// Value Constructor/Destructor
Value* newValue(double value, Value* ancestors[], int ancestorArrLen, char opString[]);
void freeValue(Value** v);

// Value Operations
void addBackward(Value* v);
Value* Add(Value* a, Value* b, GraphStack* graphStack);

void mulBackward(Value* v);
Value* Mul(Value* a, Value* b, GraphStack* graphStack);

void reluBackward(Value* v);
Value* ReLU(Value* a, GraphStack* graphStack);

// zero gradients
void ZeroGrad(MLP* mlp);