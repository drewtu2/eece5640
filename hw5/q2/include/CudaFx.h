#ifndef __CUDAFX_H__
#define __CUDAFX_H__

#include <vector>

using std::vector;

void cuda_run(vector<int>* num, vector<int>* results, int num_classes, int max_num);
__global__ void bin(int* out, int* in, int size_input, int num_classes, int max_num);
__device__ int value_to_bin(int value, int max_num, int num_classes);


#endif
