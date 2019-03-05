#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "MonteCarloPi.h"

typedef std::chrono::high_resolution_clock Clock;


using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
    // initialize MPI  
    MPI_Init(&argc,&argv);

    // Every rank will call this
    MonteCarloPi mcp(MPI_COMM_WORLD, 1000000);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto t1 = Clock::now();

    if(rank == 0) {
        auto t1 = Clock::now();
    }

    mcp.run();
    
    if(rank == 0) {
        auto t2 = Clock::now();
        cout << "Time to find: " 
             << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
             << " ms" << std::endl;
    }

    // done with MPI  
    MPI_Finalize();

}
