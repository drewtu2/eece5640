#ifndef __TASKCHECKER__
#define __TASKCHECKER__

#include <vector>
#include <semaphore.h>

#include "Task.h"

using std::vector;

class TaskChecker: public Task {

 private:
  int id;
  int lower_bound;
  int upper_bound;
  vector<int> divisors;
  int* total;
  sem_t* mutex;



 public:
  TaskChecker(int id, int lower_bound, int upper_bound, vector<int> divisors, 
          int* total, sem_t* mutex);
  void* execute(void* args);


};


#endif
