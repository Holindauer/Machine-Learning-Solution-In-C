#include "libraries.h"
#include "macros.h"
#include "structs.h"    

/**
 * @notice loadData.c loads each row of  data from the iris.csv file into a features and targets 2D arrays
 * of Value nodes. 
 * @dev the targets are one hot encoded
 * @note This loadData.c is specific to the iris.csv dataset and will need to be modified for other datasets.
 * @param train and test are static arrays of Value nodes. For larger datasets these should be dynamically allocated.
*/
void loadData(Value* features[][IRIS_FEATURES], Value* targets[][IRIS_CLASSES]){

    // open the file
    FILE* file = fopen("data/Iris.csv", "r");
    assert(file != NULL);

    // read the file
    int rowsLoaded = 0;
    char readStr[100];

    // skip the header row
    fgets(readStr, 100, file);

    while(fgets(readStr, 100, file) != NULL){

        // remove idx column
        strtok(readStr, ",");

        // create newValues for each feature and target
        features[rowsLoaded][0] = newValue(atof(strtok(NULL, ",")), NULL, NO_ANCESTORS, "feature");
        features[rowsLoaded][1] = newValue(atof(strtok(NULL, ",")), NULL, NO_ANCESTORS, "feature");
        features[rowsLoaded][2] = newValue(atof(strtok(NULL, ",")), NULL, NO_ANCESTORS, "feature");
        features[rowsLoaded][3] = newValue(atof(strtok(NULL, ",")), NULL, NO_ANCESTORS, "feature");

        // convert the class (string) to a one-hot encoded target
        char classStr[100] = "";
        strcpy(classStr, strtok(NULL, ","));

        // @note it's probably not a good idea to create values first then add the 1 hot encoding to 
        // that empty vector because it may impact the computational graph during backpropagation
        if (strcmp(classStr, "Iris-setosa\n") == 0){
            targets[rowsLoaded][0] = newValue(1, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][1] = newValue(0, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][2] = newValue(0, NULL, NO_ANCESTORS, "target");
        }else if (strcmp(classStr, "Iris-versicolor\n") == 0){
            targets[rowsLoaded][0] = newValue(0, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][1] = newValue(1, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][2] = newValue(0, NULL, NO_ANCESTORS, "target");
        }else if (strcmp(classStr, "Iris-virginica\n") == 0){
            targets[rowsLoaded][0] = newValue(0, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][1] = newValue(0, NULL, NO_ANCESTORS, "target");
            targets[rowsLoaded][2] = newValue(1, NULL, NO_ANCESTORS, "target");
        }else{
            printf("Error: classStr not recognized\n");
            exit(1);
        }

        rowsLoaded++;
    }

    // close the file
    fclose(file);
}


/**
 * @notice freeDataFeatures() is a helper function to free the memory allocated for the features data arrays
 * @param dataArr is a 2D array of Value node ptrs
 * @param numRows is the number of rows in the dataArr 
*/
void freeDataFeatures(Value* dataArr[][IRIS_FEATURES], int numRows){
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < IRIS_FEATURES; j++){
            free(dataArr[i][j]);
        }
    }
}

/**
 * @notice freeDataTargets() is a helper function to free the memory allocated for the targets data arrays
 * @param dataArr is a 2D array of Value node ptrs
 * @param numRows is the number of rows in the dataArr 
*/
void freeDataTargets(Value* dataArr[][IRIS_CLASSES], int numRows){
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < IRIS_CLASSES; j++){
            free(dataArr[i][j]);
        }
    }
}