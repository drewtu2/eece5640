#ifndef __MONTECARLOPI_H__
#define __MONTECARLOPI_H__

#include <mpi.h>

class MonteCarloPi {
  private:
    MPI_Comm comm;
    int rank;
    int num_procs;

    int num_throws;
    int num_success;
    int global_success;
    
    void runCoordinator();
    void runThrower();

    void throwDart();
    double randZeroToOne();
    bool checkSuccess(float x, float y);

    void computePi();

  public:
    MonteCarloPi(MPI_Comm comm, int num_throws);
    void run();

};

#endif
