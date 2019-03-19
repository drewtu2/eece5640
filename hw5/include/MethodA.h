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
 public:
  MethodA(MPI_COMM comm, vector<int> numbers, int num_classes);
  void run();
}


#endif