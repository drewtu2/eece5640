#include "Task.h"
#include "MethodA.h"
#include <algorithm>    // std::min


MethodA::MethodA(MPI_COMM comm, vector<int> numbers, int num_classes) {
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
                              numbers.size());

}

void MethodA::run() {

}
