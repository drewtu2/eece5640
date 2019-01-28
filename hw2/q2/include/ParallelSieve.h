#ifndef __PARALLEL_SIEVE
#define __PARALLEL_SIEVE

#include <iostream>
#include <vector>
#include <math.h>
#include <pthread.h>
#include "Task.h"

// Following Parallel Sieve from following link
// http://www.massey.ac.nz/~mjjohnso/notes/59735/seminars/01077635.pdf
#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id)+1)-BLOCK_LOW(id)) 
#define BLOCK_OWNER(index,p,n) (((p)*(index)+1)-1)/(n)

using std::vector;

class ParallelSieve : public Task {
  private:
    int thread_id;                      // The thread this task is running on
    int num_threads;                    // Total number of threads being used
    int n;                              // Find all primes up to n
    vector<bool>* primes;               // Reference to the shared prime vector
    int* k;                             // Reference to the shaared k int
    pthread_barrier_t* b1;              // Reference to the first barrier
    pthread_barrier_t* b2;              // Reference to the second barrier

    /**
     * Returns the next k to be used
     * @current: the current index for k
     */
    int next_k(int current);

    /**
     * Set a range from low to high to a given state. 
     *
     * @low: the low index
     * @high: the high index
     * @state: a boolean to set to
     */
    void set_range(int low, int high, bool state);

    /**
     * Set all numbers on the range [low, high] to false if they are divisible
     * by a given k.
     * @k: the k to use
     * @low: the low index
     * @high: the high index
     */
    void mark_composites(int k, int low, int high);

  public:

    ParallelSieve (
            int id, 
            int num_threads,
            int n, 
            vector<bool>* primes,
            int* k,
            pthread_barrier_t* barrier1, 
            pthread_barrier_t* barrier2);

    /**
     * Worker method for the nth prime function
     * @arg: PrimeJob as arguments cast as void
     */
    void* execute(void* args);

};

#endif
