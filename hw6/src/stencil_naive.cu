#include "common.h"
#include "stencil_naive.h"
#include <stdio.h>

using namespace naive;

void naive::launch_naive_stencil(float* input_a, float* input_b, int side_size) {

    float *cuda_a, *cuda_b;
    
    // Size is cubed because a and b are n^3 memory buffers
    int bytes_in = sizeof(float) * side_size * side_size * side_size;

    // allocate the space on the GPU device
    checkCuda(cudaMalloc(&(cuda_a), bytes_in));
    checkCuda(cudaMalloc(&(cuda_b), bytes_in));

    // copy/set the memory 
    checkCuda(cudaMemset(cuda_a, 0, bytes_in));
    checkCuda(cudaMemcpy(cuda_b, input_b, bytes_in, cudaMemcpyHostToDevice));


    int xGrid = side_size / X_TILE_SIZE;
    int yGrid = side_size / Y_TILE_SIZE;
    int zGrid = side_size / Z_TILE_SIZE;

    dim3 dimBlock(X_TILE_SIZE, Y_TILE_SIZE, Z_TILE_SIZE);
    dim3 dimGrid(xGrid, yGrid, zGrid);

    printf("side size: %d\n", side_size);
    printf("tile size: %d\n", X_TILE_SIZE);
    printf("tile size: %d\n", Y_TILE_SIZE);
    printf("tile size: %d\n", Z_TILE_SIZE);

    naive_stencil<<<dimGrid, dimBlock>>>(cuda_a, cuda_b, side_size);
    
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaMemcpy(input_a, cuda_a, bytes_in, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(cuda_a));
    checkCuda(cudaFree(cuda_b));
}

__global__ 
void naive::naive_stencil(float* a, float* b, int dim_size) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x < 1 || x > dim_size - 2) {
        return;
    }
    if (y < 1 || y > dim_size - 2) {
        return;
    }
    if (z < 1 || z > dim_size - 2) {
        return;
    }

    int global_idx = offset(x, y, z, dim_size);

    a[global_idx] = .8 * 
        (b[offset(x - 1, y, z, dim_size)] + b[offset(x + 1, y, z, dim_size)]
         + b[offset(x, y - 1, z, dim_size)] + b[offset(x, y + 1, z, dim_size)]
         + b[offset(x, y, z - 1, dim_size)] + b[offset(x, y, z + 1, dim_size)]);

}

__device__ 
int naive::offset(int z, int y, int x, int dim_size) {
    int offset = (z * dim_size * dim_size)
        + (y * dim_size)
        + x;
    return offset;
}

