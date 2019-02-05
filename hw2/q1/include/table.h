#ifndef __TABLE_H
#define __TABLE_H

#include <vector>
#include <pthread.h>
#include "Task.h"

#define NORMAL 0
#define SMART 1
#define SMARTFORK 2


using std::vector;

class Table {

  private:
    int num_philosophers;
    int forum;  // This represents which philospher can take the fork
    vector<pthread_t> philosophers;
    vector<pthread_mutex_t> forks;
    pthread_barrier_t barrier;
    pthread_mutex_t lock;
    pthread_mutex_t middle_fork;
    int complete;

  public:
    /**
     * Creates a table objec twith a given number of philsophers sitting around 
     * it. 
     * @num_philosophers: the number of philosphers to use
     */
    Table(int num_philosophers);
    ~Table();

    /**
     * Run the simulation of the philsophers eating
     */
    void run(int type);

    /**
     * Helper to run each thread task
     */
    static void* thread_helper(void* args);

    /**
     *  Factory method for making a philsopher. 
     *  type: 
     *  0: normal
     *  1: smart
     */
    Task* philosopher_factory(int type, int id, int num, int* forum, vector<pthread_mutex_t>* forks);




};
#endif
