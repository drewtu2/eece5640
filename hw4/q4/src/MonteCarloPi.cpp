#include <mpi.h>
#include <iostream>

#include "MonteCarloPi.h"

using std::cout;
using std::endl;


MonteCarloPi::MonteCarloPi(MPI_Comm comm, int num_throws) {
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->num_procs);
    this->comm = comm;
    this->num_throws = num_throws;
    this->num_success = 0;
    
    srand (time(NULL) + this->rank);
}

void MonteCarloPi::run() {
    if(this->rank == 0) {
        this->runCoordinator();
    } else {
        this->runThrower();
    }

    MPI_Reduce(&this->num_success, &this->global_success, 1, MPI_INT, MPI_SUM, 0,
            MPI_COMM_WORLD);
    if(this->rank == 0) {
        this->computePi();
    }
}

void MonteCarloPi::runCoordinator() {
    cout << "No real work to be done on rank 0... going to run a thrower" << endl;
    cout << this->num_procs << " total workers..." << endl;
    this->runThrower();
}

void MonteCarloPi::runThrower() {
    for(int ii = 0; ii < this->num_throws; ++ii) {
        this->throwDart();
    }
    cout << "Process " << this->rank << " threw " << this->num_success 
        << " successes!" << endl;
}

void MonteCarloPi::throwDart() {
    double x = this->randZeroToOne();
    double y = this->randZeroToOne();

    // Make sure we got a good measurement
    if(this->checkSuccess(x, y)) {
        this->num_success++;
    }

}

bool MonteCarloPi::checkSuccess(float x, float y) {
    return (x*x) + (y*y) <= 1;
}

double MonteCarloPi::randZeroToOne() {
    return rand() / (RAND_MAX + 1.);
}

void MonteCarloPi::computePi() {
    double num_throws = this->num_throws * this->num_procs;
    double pi = 4.0*double(this->global_success)/num_throws;

    cout << "PI: " << pi << " computed from " << num_throws << " throws" << endl;
}


