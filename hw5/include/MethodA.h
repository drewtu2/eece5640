#ifndef __MethodA_H__
#define __MethodA_H__

#include <vector>
#include <mpi.h>

#include "Task.h"

class MethodA : public Task {
 private:
  int num_classes;
  int first_index;
  int last_index;
  vector<int> bins;
  vector<int> results;

  int value_to_bin(int value);
 public:
  MethodA(MPI_Comm comm, vector<int> numbers, int num_classes);
  void run();
  void print_results();
};


#endif
