#ifndef __PRIMEJOB_H
#define __PRIMEJOB_H

#include <iostream>
#include <pthread.h>
#include <vector>
#include "FindPrimes.h"

using std::vector;

/**
 * Represents a prime job
 *
 */
class PrimeJob {
  public:
    FindPrimes* instance;
    int thread_id;
    int num_threads;
    int n;
    pthread_barrier_t* b1;
    pthread_barrier_t* b2;

    /**
     * Construct a PrimeJob
     * @id: thread id
     */
    PrimeJob (FindPrimes* instance, 
            int id, 
            int num_threads,
            int n, 
            pthread_barrier_t* barrier1, 
            pthread_barrier_t* barrier2) {
        this->instance = instance;
        this->thread_id = id;
        this->num_threads = num_threads;
        this->n = n;
        this->b1 = barrier1;
        this->b2 = barrier2;
    }
};

#endif
