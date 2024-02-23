#pragma once

// standard libs
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

// header files
#include "value.h"
#include "autoGrad.h"
#include "graphStack.h"
#include "hashTable.h"
#include "backward.h"

// macros
#define NO_ANCESTORS 0
#define BINARY 1
#define UNARY 0
#define HASHTABLE_SIZE 150