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

Task* TaskCreate(bool method, vector<int> numbers, MPI_Comm comm) {
  if(method) {
    int num_classes;
    cout << "\nNumber of classes? ";
    cin >> num_classes;
    cout << endl;
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
    numbers[index] = rand() % MAX_NUM + 1;
  }

  return numbers;
}

int main(int argc, char *argv[]) {
  bool method;
  int size;

  //cout << "Method? A = 0, B = 1: ";
  //cin >>  method;
  //cout << endl << "How many numbers?: ";
  //cin >>  size;
  method = 0;
  size = 1000;

  vector<int> numbers = generateNumbers(size);

  MPI_Init(&argc, &argv);
  
  Task* histogram = TaskCreate(method, numbers, MPI_COMM_WORLD);
  histogram->run();

  MPI_Finalize();

  return 0;
}
