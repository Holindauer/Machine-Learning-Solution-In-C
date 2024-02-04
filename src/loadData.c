// #include "libraries.h"
// #include "macros.h"
// #include "autoGrad.c"


// /**
//  * @notice loadData.c loads each row of  data from the iris.csv file into train and test 2D arrays
//  * of Value nodes. 
// */


// Value* loadData(Value** train, Value** test){

//     // open the file
//     FILE* file = fopen(DATA_FNAME, "r");
//     assert(file != NULL);

//     // allocate memory for the train and test arrays
//     Value** train = (Value**)malloc(sizeof(Value*) * IRIS_ROWS * (IRIS_COLS-1));
//     assert(train != NULL);

//     Value** test = (Value**)malloc(sizeof(Value*) * IRIS_ROWS);
//     assert(test != NULL);

//     free(train);
//     free(test);

//     // read the file
//     char tempStr[100];
//     while(fgets(tempStr, 100, file) != NULL){
//         printf("%s", tempStr);
//     }




//     // close the file
//     fclose(file);

//     return train[0];


// }