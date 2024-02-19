#include "autoGrad.h"
#include "lib.h"

//---------------------------------------------------------------------------------------------------------------------- Value Constructor

/**
 * @note newValue() allocates memory for a Value struct and initializes its fields given passed arguments
 * @param value double to set the value ofthe Value struct to. 
 * @param ancestors optional array of ptrs to ancestor Value structs that created this Value
 * @param ancestorsArrLen The integer number of ancestors in the array
 * @param opString A string identifying the operation that created this node
 * @return A ptr to the newly created Value struct.
*/
Value* newValue(double value, Value* ancestors[], int ancestorArrLen, char opString[]){

    // allocate mem 
    Value* v = (Value*)malloc(sizeof(Value));
    assert(v != NULL);

    // init value fields
    v->value = value;
    v->grad = 0;
    v->ancestorArrLen = ancestorArrLen;

    // New Node (not created from an operation)
    if (ancestorArrLen == NO_ANCESTORS && ancestors == NULL){
        v->ancestors == NULL;
    }
    // New Node (derived from existing nodes via an operation)
    else if (ancestorArrLen > 0 && ancestors != NULL){  

        // allocate mem for ptrs to ancestors
        v->ancestors = (Value**)malloc(ancestorArrLen * sizeof(Value*));
        assert(v->ancestors != NULL);

        // link to ancestor nodes 
        for (int i = 0; i < ancestorArrLen; i++){

            assert(ancestors[i] != NULL);
            v->ancestors[i] = ancestors[i];
        }
    }else{
        printf("Unexpected Behavior in newValue related to graph ancestors");
        exit(0);
    }

    // Backward ptr set to NULL
    v->Backward = NULL;

    // allocate mem for operation str, then copy 
    v->opString = (char*)malloc(strlen(opString) + 1); // +1 for \n 
    assert(v->opString != NULL);
    strcpy(v->opString, opString);

    return v;
}

//---------------------------------------------------------------------------------------------------------------------- Value Destructor

/**
 * @note freeValue is used to free the memory within a value struct
 * @dev all ptrs are set to NULL after releasing
 * @param v ptr to a ptr to a Value to free
*/
void freeValue(Value** v){
   
    assert((*v) != NULL);

    // free dynamically allocated members first
    if ((*v)->ancestors != NULL){

        free((*v)->ancestors); 
        (*v)->ancestors = NULL;
    }
    if ((*v)->opString != NULL){

        free((*v)->opString);
        (*v)->ancestors = NULL;
    }
       
    // free Value struct itself and set to NULL
    free((*v));
    (*v) = NULL;
}


//---------------------------------------------------------------------------------------------------------------------- Add Operation

