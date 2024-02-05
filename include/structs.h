#pragma once
#include "libraries.h"
#include "macros.h"

// structs.h


/**
 * @notice pBackwardFunc is a pointer to a function that computes the derivative of the operation 
 * that produced a given Value struct based on it's ancestor values.
 * @param Value* The value struct to compute the derivative for based on it's ancestors.
*/
typedef struct _value Value; // <--- forward declaration for self-reference in pBackwardFunc
typedef void (*pBackwardFunc)(Value*);


/**
 * @notice Value is the central struct in the autoGrad.c implementation. It represents a single value in the
 * computational graph as it passes through the network. The graph is constructed as operations are performed.
 * @param value The value of the node
 * @param grad The gradient of the node
 * @param Backward ptr to a d/dx function for the operation that produced the value
 * @param ancestors arr of ancestor nodes (dynamically allocated)
 * @param op Str of operation that produced the value (debugging)
*/
typedef struct _value {
    double value;             
    double grad;              
    pBackwardFunc Backward; 
    Value **ancestors;       
    char* opStr;                 
} Value;

/**
 * @notice VisitedNode is a struct that represents a node in the hash table
 * @param key The value of the node
 * @param next A pointer to the next node in the list
*/
typedef struct VisitedNode {
    Value* key;
    struct VisitedNode* next;
} VisitedNode;

/**
 * @notice HashTable is a struct that represents a hash table
 * @param buckets An array of pointers to VisitedNode structs
 * @param size The size of the hash table
*/
typedef struct {
    VisitedNode** buckets;
    int size;
} HashTable;


// Value Related Prototypes
Value* newValue(double _value, Value* _ancestors[], int _ancestorArrLen, char _opStr[]);
void addBackward(Value* v);
Value* Add(Value* a, Value* b);
void mulBackward(Value* v);
Value* Mul(Value* a, Value* b);
void reluBackward(Value* v);
Value* ReLU(Value* a);
void dfs(Value* v, HashTable* visited, Value*** stack, int* index);
void reverseArray(Value** arr, int start, int end);
void reverseTopologicalSort(Value* start, Value*** sorted, int* count);
void Backward(Value* v);
void freeValue(Value* v);

// Hash Table Related Prototypes
HashTable* createHashTable(int size);
unsigned int hashValuePtr(void* ptr, int size);
void insertVisited(HashTable* table, Value* value);
bool isVisited(HashTable* table, Value* value);
void freeHashTable(HashTable* table);

// Load Data Related Prototypes
void loadData(Value* features[][IRIS_FEATURES], Value* targets[][IRIS_CLASSES]);
void freeDataFeatures(Value* dataArr[][IRIS_FEATURES], int numRows);
void freeDataTargets(Value* dataArr[][IRIS_CLASSES], int numRows);