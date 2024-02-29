#include "lib.h"
#include "loadData.h"

int main(void){

    printf("\nLoading Data...\n");

    Dataset* dataset = loadData();

    printf("\nLoaded Data:");
    for (int i=0; i<NUM_EXAMPLES; i++){
        printf("\nFeature %d: %lf %lf %lf %lf", i, dataset->features[i][0]->value, dataset->features[i][1]->value, dataset->features[i][2]->value, dataset->features[i][3]->value);
        printf(" ---- Target: %lf %lf %lf", dataset->targets[i][0]->value, dataset->targets[i][1]->value, dataset->targets[i][2]->value);
    }

    freeDataset(&dataset);
    assert(dataset == NULL);

    return 0;
}