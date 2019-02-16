#include <iostream>
#include <vector>

#include "TaskChecker.h"

using std::vector;
using std::cout;
using std::endl;

TaskChecker::TaskChecker(int id, int lower_bound, int upper_bound, 
                         vector<int> divisors) {
  this->id = id;
  this->lower_bound = lower_bound;
  this->upper_bound = upper_bound;
  this->divisors = divisors;
}


void* TaskChecker::execute(void* args) {
  int count_divisible = 0;

  for(int ii = this->lower_bound; ii < this->upper_bound; ++ii) {
    for(vector<int>::iterator iter = this->divisors.begin(); 
        iter != this->divisors.end(); 
        ++iter) {

      if(ii % *iter == 0) {
        count_divisible++;
        cout << "Thread " << this->id << ii << " is divisible by " << *iter << endl;
        break;
      }
    }
  }
}
