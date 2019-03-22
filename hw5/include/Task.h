#ifndef __TASK_H__
#define __TASK_H__

#include <vector>
#include <mpi.h>


using std::vector;


class Task {
 protected:
  vector<int> nums; 
  MPI_Comm comm;
  int comm_rank;
  int comm_size;
  int max_num;

 public:
  virtual void run() = 0;
  virtual void print_results() = 0;


};


#endif
