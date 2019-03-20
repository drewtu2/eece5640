#include "Task.h"
#include "MethodB.h"
#include <algorithm>    // std::min
#include <iostream> 

using std::cout;
using std::endl;


MethodB::MethodB(MPI_COMM comm, vector<int> numbers)
{
  this->comm = comm;
  this->max_num = 1000;
  this->nums = numbers;
  this->count = 0;

  MPI_Comm_rank(this->comm, this->comm_rank);
  MPI_Comm_rank(this->comm, this->comm_size);

  this->num_classes = this->comm_size;

  this->results = vector<int>(this->num_classes);
}

void MethodB::run() {
  for(int ii = 0; ii < this->nums.size(); ++ii) {
    if(value_in_bin(nums[ii])) {
      cout << "Value: " << nums[ii] << " \t bin: " << this->comm_size<< endl;
      count++;
    }
  }

  MPI_Gather(&this->count, 1, MPI_INT, this->results.data(), 1, MPI_INT, 0, this->comm);

  if(this->comm_rank == 0) {
    for(auto it = this->results.begin(); it != this->results.end(); ++it) {
      cout << *it << endl;
    }
  }
}

bool MethodB::value_in_bin(int value) {
  int class_size = this->max_num / this->num_classes; // The size of each class
  int bin_min, bin_max;

  bin_min = this->comm_rank * class_size;
  bin_max = (this->comm_rank + 1) * class_size;

  return bin_min <= value && value <= bin_max;
}
