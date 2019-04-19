#ifndef __stencil_naive_h
#define __stencil_naive_h

namespace naive {
    void launch_naive_stencil(float* input_a, float* input_b, int side_size);
    __global__ void naive_stencil(float* a, float* b, int dim_size);
    __device__ int offset(int z, int y, int x, int dim_size);
}

#endif
