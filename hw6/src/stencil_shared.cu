#include "stencil_shared.h"
#include "common.h"
#include <stdio.h>

using namespace shared;

void shared::launch_shared_stencil(float* input_a, float* input_b, int side_size) {

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
    
    int shared_block_size = (X_TILE_SIZE + (2*RADIUS)) 
        * (Y_TILE_SIZE + (2*RADIUS)) 
        * (Z_TILE_SIZE + (2*RADIUS)) *sizeof(float);

    shared_stencil<<<dimGrid, dimBlock, shared_block_size>>> (cuda_a, cuda_b, side_size);
    
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaMemcpy(input_a, cuda_a, bytes_in, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(cuda_a));
    checkCuda(cudaFree(cuda_b));
}

__device__ 
int shared::l_offset(int x, int y, int z, int radius) {

    int dimX = 2*radius + blockDim.x;
    int dimY = 2*radius + blockDim.y;


    int offset = 
        (z + radius) * dimX * dimY    // offset to slice 
        + (y + radius) * dimX         // offset to row
        + x + radius;                 // offset to col (place in row)

    int shared_block_size = (X_TILE_SIZE + (2*RADIUS)) 
        * (Y_TILE_SIZE + (2*RADIUS)) 
        * (Z_TILE_SIZE + (2*RADIUS));
    
    if (offset < 0 || offset > shared_block_size) {
        printf("Bad local offset requested: %d Max: %d\n"
                "Requested: %d %d %d\n"
                "Dims: %d %d %d\n", 
                offset, shared_block_size, 
                x, y, z,
                dimX, dimY, blockDim.z);
    }

    return offset;
}

__device__ 
int shared::offset(int x, int y, int z, int dim_size) {
    int offset = (x * dim_size * dim_size)
        + (y * dim_size)
        + z;
    
    int shared_block_size = dim_size*dim_size*dim_size;
    if (offset < 0 || offset > shared_block_size) {
        printf("Bad global offset requested: %d Max: %d\n"
                "Requested: %d %d %d\n"
                "Dims: %d %d %d %d\n",
                offset, shared_block_size,
                x, y, z,
                gridDim.x, blockIdx.x, blockDim.x, threadIdx.x);
    }
    
    return offset;
}

__global__ 
void shared::shared_stencil(float* a, float* b, int dim_size) {

    int radius = RADIUS;
    
    extern __shared__ float s[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    
    int global_idx = offset(x, y, z, dim_size);
    // Calcualte indices
    int local_idx = l_offset(threadIdx.x, threadIdx.y, threadIdx.z, radius);
    
    // Everyone
    s[local_idx] = b[global_idx];
    
    // Break out for threads that are t the global edges
    if (x < 1 || x > dim_size - 2) {
        return;
    }
    if (y < 1 || y > dim_size - 2) {
        return;
    }
    if (z < 1 || z > dim_size - 2) {
        return;
    }
    
    if(threadIdx.x < radius) {
        int l_idx1 = l_offset(threadIdx.x - radius, threadIdx.y, threadIdx.z, radius);
        int l_idx2 = l_offset(threadIdx.x + blockDim.x, threadIdx.y, threadIdx.z, radius);
        int g_idx1 = offset(x - radius, y, z, dim_size);

        s[l_idx1] = b[g_idx1];

        if(blockIdx.x < gridDim.x - 1) {
            int g_idx2 = offset(x + blockDim.x, y, z, dim_size); // wrong
            s[l_idx2] = b[g_idx2];
        }
    }
    
    if(threadIdx.y < radius) {
        int l_idx1 = l_offset(threadIdx.x, threadIdx.y - radius, threadIdx.z, radius);
        int l_idx2 = l_offset(threadIdx.x, threadIdx.y + blockDim.y, threadIdx.z, radius);
        int g_idx1 = offset(x, y - radius, z, dim_size);
        
        s[l_idx1] = b[g_idx1];
        
        if(blockIdx.y < gridDim.y - 1) {
            int g_idx2 = offset(x, y + blockDim.y , z, dim_size); // correct
            s[l_idx2] = b[g_idx2];
        }
    }
    
    if(threadIdx.z < radius) {
        int l_idx1 = l_offset(threadIdx.x, threadIdx.y, threadIdx.z - radius, radius);
        int l_idx2 = l_offset(threadIdx.x, threadIdx.y, threadIdx.z + blockDim.z, radius);
        int g_idx1 = offset(x, y, z - radius, dim_size);
        
        s[l_idx1] = b[g_idx1];
        
        if(blockIdx.z < gridDim.z - 1) {
            int g_idx2 = offset(x, y, z + blockDim.z, dim_size);
            s[l_idx2] = b[g_idx2];
        }
    }

    // sync here...
    __syncthreads();

    a[global_idx] = .8 * 
         ( s[l_offset(threadIdx.x - 1, threadIdx.y, threadIdx.z, radius)] 
         + s[l_offset(threadIdx.x + 1, threadIdx.y, threadIdx.z, radius)]
         + s[l_offset(threadIdx.x, threadIdx.y - 1, threadIdx.z, radius)] 
         + s[l_offset(threadIdx.x, threadIdx.y + 1, threadIdx.z, radius)]
         + s[l_offset(threadIdx.x, threadIdx.y, threadIdx.z - 1, radius)] 
         + s[l_offset(threadIdx.x, threadIdx.y, threadIdx.z + 1, radius)]);

    //a[global_idx] = .8 * 
    //    (b[offset(x - 1, y, z, dim_size)] + b[offset(x + 1, y, z, dim_size)]
    //     + b[offset(x, y - 1, z, dim_size)] + b[offset(x, y + 1, z, dim_size)]
    //     + b[offset(x, y, z - 1, dim_size)] + b[offset(x, y, z + 1, dim_size)]);

}

