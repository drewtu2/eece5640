// Author: Nat Tuck
// CS3650 starter code
#ifndef UTILS_H
#define UTILS_H

#include "barrier.h"
#include "float_vec.h"


void seed_rng();
void check_rv(int rv);

int compare_float(const void* a, const void* b);

typedef struct sort_job {
  int pnum;
  floats* input;
  const char* output;
  int totalProcs;
  floats* samps;
  long* sizes;
  barrier* bb;

} sort_job;

sort_job* make_sort_job(int pnum, floats* input, const char* output, int P,
                        floats* samps, long* sizes, barrier* bb);

int sum_array(long*xs, int end_index);

#endif

