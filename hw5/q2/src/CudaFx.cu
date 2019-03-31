#include <vector>
#include <iostream>
#include <chrono>

#include "CudaFx.h"

using std::vector;
using std::cout;
using std::endl;
typedef std::chrono::high_resolution_clock Clock;

__host__ void cuda_run(vector<int>* nums, vector<int>* results, int num_classes, int max_num) {
    // Calculate bytes needed for input and output
    int bytes_in = sizeof(int) * nums->size();
    int bytes_out = sizeof(int) * results->size();

    int* d_input, *d_output;

    int blockSize = 256;
    int gridSize = (int)ceil(nums->size()/(float)blockSize); //TODO: Choose these more wisely...
    
    //cout << "bytes in: " << bytes_in << endl;
    //cout << "bytes out: " << bytes_out << endl;
    //cout << "Size: " << nums->size() << endl;
    //cout << "Using " << gridSize << " grids" << endl;
    //cout << "Total Threads: " << gridSize*blockSize << endl;

    // Malloc the necessary bytes
    cudaMalloc(&(d_input), bytes_in);
    cudaMalloc(&(d_output), bytes_out);
    
    // Copy to data to device
    cudaMemcpy(d_input, nums->data(), bytes_in, cudaMemcpyHostToDevice);

    // Grid Size and block size yadda yadda
    auto t1 = Clock::now();
    bin<<<gridSize, blockSize>>>(d_output, d_input, nums->size(), num_classes, max_num);
    auto t2 = Clock::now();
    
    cout << "Time to run kernel: " 
        << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
        << " ms" << std::endl;
    
    // Copy results from device
    cudaMemcpy(results->data(), d_output, bytes_out, cudaMemcpyDeviceToHost);
}

__global__ void bin(int* out, int* in, int size_input, int num_classes, int max_num) {
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;    // Global index
    // Short circuit for extra threads
    if(thread_id >= size_input) {
        return;
    }

    int value = in[thread_id];  // Value of this thread id
    int bin_id = value_to_bin(value, max_num, num_classes);
    atomicAdd(&out[bin_id], 1);
}


/**
 * Calcualtes the appropriate bin index for a given value
 */
__device__ int value_to_bin(int value, int max_num, int num_classes) {
    int class_size = max_num / num_classes; // The size of each class
    int bin_min, bin_max;

    for(int ii = 0; ii < num_classes; ++ii) {
        bin_min = ii * class_size;
        bin_max = (ii + 1) * class_size;

        if(bin_min <= value && value < bin_max) {
            return ii;
        }
    }
    return 0;
}
