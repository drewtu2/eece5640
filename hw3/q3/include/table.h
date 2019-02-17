#ifndef __TABLE_H
#define __TABLE_H

#include <vector>
#include <omp.h>
#include "Task.h"

#define NORMAL 0
#define SMART 1
#define SMARTFORK 2


using std::vector;

class Table {

  private:
    int num_philosophers;
    int forum;  // This represents which philospher can take the fork
    vector<omp_lock_t> forks;
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
     *  Factory method for making a philsopher. 
     *  type: 
     *  0: normal
     *  1: smart
     */
    Task* philosopher_factory(int type, int* forum, vector<omp_lock_t>* forks);




};
#endif
