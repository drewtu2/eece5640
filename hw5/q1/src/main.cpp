#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>

#include "Task.h"
#include "MethodA.h"
#include "MethodB.h"

#define MAX_NUM 1000

using std::vector;
using std::cout;
using std::cin;
using std::endl;

typedef struct config {
    bool method;
    int size;
    int num_classes;
} config;

Task* TaskCreate(int method, vector<int> numbers, MPI_Comm comm, int num_classes) {
    // If method == 1, return A
    if(method) {
        return new MethodA(comm, numbers, num_classes);
    }

    return new MethodB(comm, numbers);
}

/**
 * Generates a vector of N random ints
 */
vector<int> generateNumbers(int N) {
    vector<int> numbers(N);

    for(int index = 0; index < numbers.size(); ++index) {
        numbers[index] = rand() % MAX_NUM;
    }

    return numbers;
}

Task* init() {
    int method;
    int rank;
    int size;
    int num_classes;
    vector<int> numbers;
    srand(time(NULL));

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Collect input on rank 0
    if(rank == 0) {
        cout << "Method? A = 1, B = 0: ";
        cin >>  method;
        cout << endl << "How many numbers?: ";
        cin >>  size;

        if(method) {
            cout << "\nNumber of classes? ";
            cin >> num_classes;
            cout << endl;
        }
        numbers = generateNumbers(size);
    } 

    // Broadcast stuff so that all ranks...
    MPI_Bcast(&method, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the vectors on all of the other ranks..
    if(rank != 0) {
        numbers.resize(size);
    }

    // Send The generated numbers
    MPI_Bcast(&numbers.front(), numbers.size(), MPI_INT, 0, MPI_COMM_WORLD);

    return TaskCreate(method, numbers, MPI_COMM_WORLD, num_classes);
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    Task* histogram = init();
    histogram->run();
    histogram->print_results();
    MPI_Finalize();

    return 0;
}
