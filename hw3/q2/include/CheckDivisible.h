#ifndef __CHECKDIVISIBLE__
#define __CHECKDIVISIBLE__

#include <vector>
#include "pthread.h"

using std::vector;

class CheckDivisible {
 private:
  int num_threads;
  int max;
  vector<pthread_t> threads;
  vector<int> divisors;

 public:
  CheckDivisible();
  CheckDivisible(int num_threads, int max, vector<int> divisors);
  ~CheckDivisible();

  void run();
  void* thread_helper(void* args);

};

#endif
