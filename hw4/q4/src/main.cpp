#include <mpi.h>
#include <iostream>
#include <stdio.h>

#include "MonteCarloPi.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
	int  numtasks, rank, len, rc; 
	char hostname[MPI_MAX_PROCESSOR_NAME];

	// initialize MPI  
	MPI_Init(&argc,&argv);

	MonteCarloPi mcp(MPI_COMM_WORLD, 10);
	mcp.run();

	// do some work with message passing 


	// done with MPI  
	MPI_Finalize();
}
