#ifndef __TABLE_H
#define __TABLE_H

#include <vector>
#include <pthread.h>

using std::vector;

class Table {

  private:
    int num_philosophers;
    int forum;  // This represents which philospher can take the fork
    vector<pthread_t> philosophers;
    vector<pthread_mutex_t> forks;

  public:
    /**
     * Creates a table objec twith a given number of philsophers sitting around 
     * it. 
     * @num_philosophers: the number of philosphers to use
     */
    Table(int num_philosophers);

    /**
     * Run the simulation of the philsophers eating
     */
    void run();

    /**
     * Helper to run each thread task
     */
    static void* thread_helper(void* args);




};
#endif
