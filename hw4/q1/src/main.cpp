#include <iostream>
#include <vector>

#ifdef OMP
#include <omp.h>
#endif

#include "EigenInverse.h"
#include "ParallelGaussian.h"

using std::cout;
using std::endl;

float** test_a() {
  float base[9] = {1, 2, 1, 2, 1, 0, -1, 1, 2};

  float** arr = new float*[3];
  for(int row = 0; row < 3; ++row) {
    arr[row] = new float[row]();
    
    for(int col = 0; col < 3; ++col) {
      arr[row][col] = base[row*3 + col];
    }
  }
  
  return arr;
}

int main()
{
#ifdef OMP
  omp_set_num_threads(omp_get_max_threads());
#endif

  float** f = test_a();

  ParallelGaussian pg = ParallelGaussian(f, 3, 3);
  pg.run();


  vector<vector<float> > m = {{1, 2, 1}, {2, 1, 0}, {-1, 1, 2}};
  EigenInverse inv(m, 3, 3);
  cout << inv.run() << endl;


}
