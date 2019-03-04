#include <mpi.h>
#include <iostream>
#include <stdio.h>

#include "MonteCarloPi.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
    // initialize MPI  
    MPI_Init(&argc,&argv);

    // Every rank will call this
    MonteCarloPi mcp(MPI_COMM_WORLD, 1000000);
    mcp.run();

    // done with MPI  
    MPI_Finalize();
}
