#include "loadData.h"

// this function loads in the iris dataset csv gile in the data dir
Dataset* loadData(void){

    // allocate mem for dataset
    Dataset* dataset = malloc(sizeof(Dataset));
    assert(dataset != NULL);

    // allocate mem for an array of arrays of Value struct ptrs for features and targets
    dataset->features = malloc(sizeof(Value***) * NUM_EXAMPLES);
    dataset->targets = malloc(sizeof(Value***) * NUM_EXAMPLES);
    assert(dataset->features != NULL);
    assert(dataset->targets != NULL);

    // init all Value ptrs in features and target arrays
    for (int i=0; i<NUM_EXAMPLES; i++){

        dataset->features[i] = malloc(sizeof(Value*) * NUM_FEATURES);
        dataset->targets[i] = malloc(sizeof(Value*) * NUM_CLASSES);
        assert(dataset->features[i] != NULL);
        assert(dataset->features[i] != NULL);

    }

    // Init all Values in features and targets
    for (int example=0; example<NUM_EXAMPLES; example++){

        // init features
        for (int feature = 0; feature<NUM_FEATURES; feature++){
            dataset->features[example][feature] = newValue(0, NULL, NO_ANCESTORS, "feature");
        }

        // init examples
        for (int class = 0; class<NUM_CLASSES; class++){
            dataset->targets[example][class] = newValue(0, NULL, NO_ANCESTORS, "target");
        }
    }


    // open the file
    FILE* file = fopen("data/Iris.csv", "r");
    assert(file != NULL);

    // read the file
    int rowsLoaded = 0;
    char tempStr[100];

    while(fgets(tempStr, 100, file) != NULL){

        // remove idx column
        strtok(tempStr, ",");

        // create newValues for each feature and target
        dataset->features[rowsLoaded][0]->value = atof(strtok(NULL, ","));
        dataset->features[rowsLoaded][1]->value = atof(strtok(NULL, ","));
        dataset->features[rowsLoaded][2]->value = atof(strtok(NULL, ","));
        dataset->features[rowsLoaded][3]->value = atof(strtok(NULL, ","));

        // convert the class (string) to a one-hot encoded target
        char classStr[100] = "";
        strcpy(classStr, strtok(NULL, ","));

        printf("\nclassStr: %s", classStr);

        // @note it's probably not a good idea to create values first then add the 1 hot encoding to 
        // that empty vector because it may impact the computational graph during backpropagation
        if (strcmp(classStr, "Iris-setosa\n") == 0){
            dataset->targets[rowsLoaded][0]->value = 1;

        }else if (strcmp(classStr, "Iris-versicolor\n") == 0){
            dataset->targets[rowsLoaded][1]->value = 1;
        
        }else if (strcmp(classStr, "Iris-virginica\n") == 0 || strcmp(classStr, "Iris-virginica") == 0){ // <--- last class does not have a newline
            dataset->targets[rowsLoaded][2]->value = 1;

        }else{

            printf("Error: classStr not recognized\n");
            exit(1);
        }

        rowsLoaded++;
    }

    // close the file
    fclose(file);

    return dataset;
}

