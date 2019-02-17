#include <iostream>
#include <vector>
#include <semaphore.h>

#include "TaskChecker.h"

using std::vector;
using std::cout;
using std::endl;

TaskChecker::TaskChecker(int id, int lower_bound, int upper_bound, 
                         vector<int> divisors, int* total, sem_t* mutex) {
  this->id = id;
  this->lower_bound = lower_bound;
  this->upper_bound = upper_bound;
  this->divisors = divisors;
  this->total = total;
  this->mutex = mutex;
}


void* TaskChecker::execute(void* args) {
  int count_divisible = 0;

  for(int ii = this->lower_bound; ii < this->upper_bound; ++ii) {
    for(vector<int>::iterator iter = this->divisors.begin(); 
        iter != this->divisors.end(); 
        ++iter) {

      if(ii % *iter == 0) {
        count_divisible++;
        cout << "Thread " << this->id << " " << ii << " is divisible by " << *iter << endl;
        break;
      }
    }
  }

  // Update global value
  sem_wait(this->mutex);
  cout << "Thread " << this->id << " entered mutex." << endl;
  *this->total = *this->total + count_divisible;
  sem_post(this->mutex);
  cout << "Thread " << this->id << " exited mutex." << endl;
}
