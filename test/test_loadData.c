// #include "libraries.h"

// /**
//  * @notice test_loadData tests the loadData function by loading the iris.csv file into the train and test arrays
//  * and checking that the data was loaded correctly
// */
// void test_loadData(void){

//     // load the data
//     Value** train = (Value**)malloc(sizeof(Value*) * IRIS_ROWS * (IRIS_COLS-1));
//     assert(train != NULL);

//     Value** test = (Value**)malloc(sizeof(Value*) * IRIS_ROWS);
//     assert(test != NULL);

//     loadData(train, test);
// }