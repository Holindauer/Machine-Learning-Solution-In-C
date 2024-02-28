#pragma once
#include "value.h"
#include "mlp.h"

// forward functions
Value** newOutputVector(int outputSize);
Value** MultiplyWeights(Layer* layer, Value** input, GraphStack* graphStack);
Value** AddBias(Layer* layer, Value** input, GraphStack* graphStack);
Value** ApplyReLU(Layer* layer, Value** input, GraphStack* graphStack);
Value** Forward(MLP* mlp, Value** input);