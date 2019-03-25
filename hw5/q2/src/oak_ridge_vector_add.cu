#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vecAdd(double*a, double*b, double*c, int n) { // CUDA kernel
    int id = blockIdx.x*blockDim.x+threadIdx.x; // get global thread id
    if (id < n) { // make sure we do not go out of bounds
        c[id] = a[id] + b[id];
    }
}

int main( int argc, char* argv []) {
    int n = 100000;     // Size of vectors
    double *h_a, *h_b, *h_c;  // host input vectors
    double *d_a, *d_b;  // device input vectors
    double *d_c;        // device output vector
    size_t bytes = n*sizeof(double); // size, in bytes, of each vector
    
    h_a = (double*) malloc(bytes); 
    h_b = (double*) malloc(bytes); 
    h_c = (double*) malloc(bytes); 

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    for(int ii = 0; ii < n; ++ii) {
        h_a[ii] = sin(ii)*sin(ii);
        h_b[ii] = cos(ii)*cos(ii);
    }   
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize); // Number of thread blocks in grid

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n); // execute the kernel
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); //Copy back to host

    // Sum up vector c and print result divided by n
    double sum = 0;

    for(int ii = 0; ii < n; ++ii) {
        sum += h_c[ii];
    }

    printf("final result: %f\n", sum/n); // the answer should be 1.0

    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;

}
