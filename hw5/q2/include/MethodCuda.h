#ifndef __METHOD_CUDA_H__
#define __METHOD_CUDA_H__

#include <vector>
#include "Task.h"

class MethodCuda : public Task {
    private:
        int* d_input;
        int* d_output;
        vector<int> results;
        int num_classes;


    public:
        MethodCuda(vector<int> input, int num_classes);
        void run();
        void print_results();

};

#endif
