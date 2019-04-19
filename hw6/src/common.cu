#include "common.h"
#include <stdio.h>


cudaError_t checkCuda(cudaError_t result) {
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            exit(-1);
        }
        return result;
}
