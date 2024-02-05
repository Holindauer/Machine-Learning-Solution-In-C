#include "libraries.h"
#include "macros.h"
#include "structs.h"

/**
 * @notice test_loadData tests the loadData function by loading the iris.csv file into the train and test arrays
 * and checking that the data was loaded correctly
*/
void test_loadData(void){

    Value* features[IRIS_ROWS][IRIS_FEATURES];
    Value* targets[IRIS_ROWS][IRIS_CLASSES];

    loadData(features, targets);

    // check some values
    assert(features[0][0]->value == 5.1);
    assert(features[149][1]->value == 3.0);
    assert(targets[149][2]->value == 1.0);
    assert(targets[0][0]->value == 1.0);

    freeDataFeatures(features, IRIS_ROWS);
    freeDataTargets(targets, IRIS_ROWS);
}


int main(void){
    printf("\nRunning test_loadData...\n");

    test_loadData();
    
    printf("test_loadData passed!\n");
    
    return 0;
}