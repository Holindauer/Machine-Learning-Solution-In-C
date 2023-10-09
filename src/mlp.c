#include "libraries.h"



// Function to generate a random float between two given values
float random_float(float min, float max) {
    return (max - min) * ((float)rand() / RAND_MAX) + min;
}

// Function to populate a weight matrix using He initialization
void he_initialize(float* matrix, int rows, int cols) {
    float stddev = sqrt(2.0 / cols);  // Standard deviation for He initialization

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate a random number from a normal distribution
            // Note: This is a simple approximation using the Box-Muller transform
            float u1 = random_float(0.0, 1.0);
            float u2 = random_float(0.0, 1.0);
            float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

            // Assign the weight
            matrix[i * cols + j] = stddev * z0;
        }
    }
}
