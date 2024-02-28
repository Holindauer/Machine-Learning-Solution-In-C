#pragma once
#include "value.h"
#include "mlp.h"

// forward functions
Value** newOutputVector(int outputSize);
Value** MultiplyWeights(Layer* layer, Value** input, GraphStack* graphStack);