#include "lib.h"

#define NUM_EXAMPLES 150
#define NUM_FEATURES 4
#define NUM_CLASSES 3

// struct to hold dataset
typedef struct {
    Value*** features;
    Value*** targets;
}Dataset;

Dataset* loadData(void);