#ifndef __MethodB_H__
#define __MethodB_H__

#include <vector>
#include <mpi.h>

#include "Task.h"

class MethodB : public Task {
 private:
  int num_classes;
  int count;
  vector<int> results;
  bool value_in_bin(int value);

 public:
  MethodB(MPI_Comm comm, vector<int> numbers);
  void run();
};


#endif
