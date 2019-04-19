#ifndef __stencil_shared_h
#define __stencil_shared_h

namespace shared {
    __device__ int l_offset(int x, int y, int z, int radius);
    __device__ int offset(int x, int y, int z, int side_dim);
    void launch_shared_stencil(float* input_a, float* input_b, int side_size);
    __global__ void shared_stencil(float* a, float* b, int dim_size);
}
#endif
