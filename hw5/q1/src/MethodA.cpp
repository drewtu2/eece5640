#include "Task.h"
#include "MethodA.h"
#include <algorithm>    // std::min
#include <iostream> 

using std::cout;
using std::endl;


    MethodA::MethodA(MPI_Comm comm, vector<int> numbers, int num_classes) 
: bins(num_classes),
    results(num_classes)
{
    this->num_classes = num_classes;
    this->comm = comm;
    this->max_num = 1000;
    this->nums = numbers;

    MPI_Comm_rank(this->comm, &this->comm_rank);
    MPI_Comm_size(this->comm, &this->comm_size);

    int nums_size = numbers.size() / this->comm_size;   // # of #'s we're responsible for
    int class_size = this->max_num / this->num_classes; // The size of each class

    // First and last indices we're responsible for...
    this->first_index = this->comm_rank * nums_size;
    this->last_index = (this->comm_rank + 1) * nums_size;
    this->last_index = std::min(this->last_index, (int) numbers.size());

    if(this->comm_rank == 0) {
        cout << "Running with Method A" << endl;
    }
}

void MethodA::run() {

    for(int ii = this->first_index; ii < this->last_index; ++ii) {
        bins[value_to_bin(nums[ii])]++;
    }

    MPI_Reduce(this->bins.data(), this->results.data(), (int)this->bins.size(),
            MPI_INT, MPI_SUM, 0, this->comm);
}

void MethodA::print_results() {

    int min, max, sum=0;
    if(this->comm_rank == 0) {
        for(int idx = 0; idx < this->results.size(); ++idx) {
            min = idx * this->max_num / this->num_classes;
            max = (idx + 1) * this->max_num / this->num_classes;
            sum += results[idx];
            cout << "(" << min << ", " << max << "): " << results[idx] << endl;
        }
        cout << "Sum: " << sum << endl;
    }

}

int MethodA::value_to_bin(int value) {
    int class_size = this->max_num / this->num_classes; // The size of each class
    int bin_min, bin_max;

    for(int ii = 0; ii < this->num_classes; ++ii) {
        bin_min = ii * class_size;
        bin_max = (ii + 1) * class_size;

        if(bin_min <= value && value < bin_max) {
            return ii;
        }
    }
    std::cerr << "Uncaught case!!" << endl;
    return 0;
}
