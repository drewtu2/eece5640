#ifndef __PHILOSOPHER
#define __PHILOSOPHER

#include <iostream>
#include <vector>
#include <omp.h>

#include "Task.h"


using std::vector;

class Philosopher: public Task {
  private:
    int thread_id;                      // The thread this task is running on
    int num_philosophers;
    int* forum;                         // A shared buffer to read from
    vector<omp_lock_t>* forks;     // Point to the forks

    /**
     * Pick up the appropriate forks for this philsopher.
     * The forks we can pick up are forks thread_id and thread_id + 1. The
     * last philsopher will pick up thread_id and 0. 
     */
    void pickup_forks();

    /**
     * Putdown the appropriate forks for this philsopher.
     * The forks we can put down are forks thread_id and thread_id + 1. The
     * last philsopher will pick up thread_id and 0. 
     */
    void putdown_forks();

  public:
    Philosopher(
            int* forum, 
            vector<omp_lock_t>* forks);

    /**
     * Worker method for the philosopher
     * @arg: any args we might want...
     */
    void* execute(void* args);

};

#endif
