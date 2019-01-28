#ifndef __TABLE_H
#define __TABLE_H

#include <vector>
#include <pthreads.h>

using std::vector;

class Table {

  private:
    vector<pthread_t> philosophers;
    vector<pthread_mutex_t> forks;

  public:











};
#endif
