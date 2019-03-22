#include "Task.h"
#include "MethodB.h"
#include <algorithm>    // std::min
#include <iostream> 

using std::cout;
using std::endl;

MethodB::MethodB(MPI_Comm comm, vector<int> numbers)
{
    this->comm = comm;
    this->max_num = 1000;
    this->nums = numbers;
    this->count = 0;

    MPI_Comm_rank(this->comm, &this->comm_rank);
    MPI_Comm_size(this->comm, &this->comm_size);

    this->num_classes = this->comm_size;

    this->results = vector<int>(this->num_classes);

    if(this->comm_rank == 0) {
        cout << "Running with Method B" << endl;
    }

    int class_size = this->max_num / this->num_classes; // The size of each class

    this->bin_min = this->comm_rank * class_size;
    this->bin_max = (this->comm_rank + 1) * class_size;
}

void MethodB::run() {
    for(int ii = 0; ii < this->nums.size(); ++ii) {
        if(value_in_bin(nums[ii])) {
            count++;
        }
    }

    MPI_Gather(&this->count, 1, MPI_INT, this->results.data(), 1, MPI_INT, 0, this->comm);
}

void MethodB::print_results() {
    int min, max, sum = 0;
    if(this->comm_rank == 0) {
        for(int idx = 0; idx < this->results.size(); ++idx) {
            min = idx * this->max_num / this->num_classes;
            max = (idx + 1) * this->max_num / this->num_classes;
            sum += this->results[idx];
            cout << "(" << min << ", " << max << "): " << this->results[idx] << endl;
        }

        cout << "Sum: " << sum << endl;
    }

}

bool MethodB::value_in_bin(int value) {
    return (this->bin_min <= value) && (value < this->bin_max);
}
