#include <iostream>
#include <omp.h>

#include "Mv.h"

using std::cout;
using std::endl;

int main() {
  
  int max_threads = omp_get_max_threads();

  cout << "Max threads: "<< max_threads << endl;

  vector<vector<int>> A = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}};
  vector<int> b = {
    1, 2, 3
  };

  cout << "A * b = " << MatrixVector.mul(A, b) << endl;

  

  return 0;
}
