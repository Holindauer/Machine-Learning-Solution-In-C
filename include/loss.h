#pragma once
#include "value.h"


// loss function funcs
double* Softmax(Value** valueArr, int lenArr);
void freeSoftmax(double** softmaxArr);

void categoricalCrossEntropyBackward(Value* v, double* softmaxOutput, Value** targetsArr, int lenArr);
Value* categoricalCrossEntropy(Value** outputArr, Value** targetsArr, double* softmaxOutput, int lenArr, GraphStack* graphStack);