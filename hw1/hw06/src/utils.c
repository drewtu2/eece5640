// Author: Nat Tuck
// CS3650 starter code

#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>

#include "utils.h"
#include "barrier.h"
#include "float_vec.h"

void
seed_rng()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    long pid = getpid();
    long sec = tv.tv_sec;
    long usc = tv.tv_usec;

    srandom(pid ^ sec ^ usc);
}

void
check_rv(int rv)
{
    if (rv == -1) {
        perror("oops");
        fflush(stdout);
        fflush(stderr);
        abort();
    }
}

int compare_float(const void* a, const void* b)
{
    return (*(int*)a - *(int*)b);
}

/*
 * Just a wrapper to pass a bunch of information as an argument to our threads.
 * This object DOES NOT own the data and therfore, only the pointer to the obj
 * needs to be freed. 
 */
sort_job* make_sort_job(int pnum, floats* input, const char* output, int P, 
                floats* samps, long* sizes, barrier* bb)
{
    sort_job* my_job = malloc(sizeof(sort_job));
    my_job->pnum = pnum;
    my_job->input = input;
    my_job->output = output;
    my_job->totalProcs = P;
    my_job->samps = samps;
    my_job->sizes = sizes;
    my_job->bb = bb;
    return my_job;
}

int sum_array(long* xs, int end_index)
{
  int sum = 0;
  for(int ii = 0; ii < end_index; ++ii)
  {
    sum += xs[ii];
  }
  return sum;
}
