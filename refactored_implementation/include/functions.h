#pragma once

#include "structs.h"
#include "libraries.h"

//load_data.c
void init_dataset(Data* dataset);
void print_dataset(Data dataset);
void load_data(Data* dataset, FILE* stream);


//model.c
void init_model(Model* model);