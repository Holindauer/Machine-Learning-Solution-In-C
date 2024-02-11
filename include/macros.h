#pragma once

#define NO_ANCESTORS 0
#define MAX_GRAPH_SIZE 2000


#define DATA_FNAME "iris.csv"
#define IRIS_ROWS 150
#define IRIS_FEATURES 4
#define IRIS_CLASSES 3

#define TRAIN_EXAMPLES 120
#define TEST_EXAMPLES 30

// for indexing a matrix array at (i, j) stored as a 1D array
#define INDEX(i, j, numCols) ((i) * (numCols) + (j)) 