#ifndef __TASKCHECKER__
#define __TASKCHECKER__

#include <vector>
#include "Task.h"

using std::vector;

class TaskChecker: public Task {

 private:
  int id;
  int lower_bound;
  int upper_bound;
  vector<int> divisors;



 public:
  TaskChecker(int id, int lower_bound, int upper_bound, vector<int> divisors);
  void* execute(void* args);


};


#endif
