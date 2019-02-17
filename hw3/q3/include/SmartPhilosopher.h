#ifndef __SMARTPHILOSOPHER
#define __SMARTPHILOSOPHER

#include <iostream>
#include <vector>
#include <omp.h>

#include "Task.h"


using std::vector;

class SmartPhilosopher: public Task {
  private:
    int thread_id;                      // The thread this task is running on
    int num_philosophers;
    int* forum;                         // A shared buffer to read from
    vector<omp_lock_t>* forks;     // Point to the forks
    int* finish_count;

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

    /**
     * Returns true if it's this philosopher's time to eat!
     */
    bool time_to_eat();

    /**
     * Print an update from thread 0;
     */
    void print_update();

    /**
     * Checks if we've finished this iteration
     */
    bool check_finished(int eat_count, int max_eats);

  public:
    SmartPhilosopher(
            int* forum, 
            vector<omp_lock_t>* forks,
            int* num_complete);

    /**
     * Worker method for the philosopher
     * @arg: any args we might want...
     */
    void* execute(void* args);

};

#endif
