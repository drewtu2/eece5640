#include "Task.h"
#include "MethodA.h"
#include <algorithm>    // std::min
#include <iostream> 

using std::cout;
using std::endl;


MethodA::MethodA(MPI_COMM comm, vector<int> numbers, int num_classes) 
  : bins(num_classes),
    results(num_classes)
{
  this->num_classes = num_classes;
  this->comm = comm;
  this->max_num = 1000;
  this->nums = numbers;


  MPI_Comm_rank(this->comm, this->comm_rank);
  MPI_Comm_rank(this->comm, this->comm_size);

  int nums_size = numbers.size() / this->comm_size;   // # of #'s we're responsible for
  int class_size = this->max_num / this->num_classes; // The size of each class
  
  // First and last indices we're responsible for...
  this->first_index = this->comm_rank * nums_size;
  this->last_index = std::min(this->first_index + nums_size, 
                              (int) numbers.size());
}

void MethodA::run() {
  for(int ii = this->first_index; ii < this->last_index; ++ii) {
    bins[value_to_bin(nums[ii])]++;
  }

  MPI_Reduce(this->bins.data(), this->results.data(), (int)this->bins.size(),
               MPI_INT, MPI_SUM, 0, this->comm);

  if(this->comm_rank == 0) {
    for(auto it = this->results.begin(); it != this->results.end(); ++it) {
      cout << *it << endl;
    }
  }
}

int MethodA::value_to_bin(int value) {
  int class_size = this->max_num / this->num_classes; // The size of each class
  int bin_min, bin_max;

  for(int ii = 0; ii < this->nums.size(); ++ii) {
    bin_min = ii * class_size;
    bin_max = (ii + 1) * class_size;

    if(bin_min <= value && value <= bin_max) {
      cout << "Value: " << value << " \t bin: " << ii << endl;
      return ii;
    }
  }

  return 0;
}
