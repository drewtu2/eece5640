#include <algorithm>    // std::min
#include <iostream> 
#include <vector>

#include "Task.h"
#include "MethodCuda.h"

using std::cout;
using std::endl;
using std::vector;

extern void cuda_run(vector<int>* num, vector<int>* results, int num_classes, int max_num);

MethodCuda::MethodCuda(vector<int> numbers, int num_classes) 
{
    this->num_classes = num_classes;
    this->max_num = 1000;
    this->nums = numbers;
    this->results = vector<int>(num_classes);

}

void MethodCuda::run() {
    cuda_run(&(this->nums), &(this->results), this->num_classes, this->max_num);
}

void MethodCuda::print_results() {
    int min, max, sum=0;
    for(int idx = 0; idx < this->results.size(); ++idx) {
        min = idx * this->max_num / this->num_classes;
        max = (idx + 1) * this->max_num / this->num_classes;
        sum += results[idx];
        cout << "(" << min << ", " << max << "): " << results[idx] << endl;
    }
    cout << "Sum: " << sum << endl;
}
