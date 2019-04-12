
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>


#define TILE_SIZE 16

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
unsigned char *input_gpu;
float *output_gpu;

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

__global__ void calculate_response(unsigned char *input, 
                       float *output,
                       unsigned int height,
                       unsigned int width){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    
    float k = .04;
	
    // This represents the first and last row/column of the image. sobel needs 
    // a delta of at least one
    if(x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2) {
        return;
    }

    // calculate the offsets for all regions of interest
    int offset = (y * width) + x;
    int x_minus_one = y*width + (x - 1);
    int x_plus_one = y*width + (x + 1);
    int y_minus_one = (y - 1)*width + x;
    int y_plus_one = (y + 1)*width + x;
    
    // Calculate dx, dy
    float dx = input[x_minus_one] - input[x_plus_one];
    float dy = input[y_minus_one] - input[y_plus_one];

    // Caclualte dx2, dy2, dxy
    float ix2 = dx*dx;
    float iy2 = dy*dy;
    float ixy = dx*dy;
    
    // Harris Corner Response Matrix: 
    // [ix2, ixy;
    // ixy, iy2]
    float itrace    = ix2 + iy2;
    float idet      = (ix2*iy2) - (ixy*ixy);

    float response = abs(idet - (k * itrace * itrace));

    output[offset] = response;
}

void gpu_function (unsigned char *input, 
                   float *output,
                   unsigned int height, 
                   unsigned int width){
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	// Both are the same size (CPU/GPU).
	int size = height*width;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(float)));
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(float)));
	
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        input, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        calculate_response<<<dimGrid, dimBlock>>>(input_gpu, 
                                      output_gpu,
                                      height, 
                                      width);
                                      
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(output, 
			output_gpu, 
			size*sizeof(float), 
			cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
}
