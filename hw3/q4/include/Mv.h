#ifndef __MATRIX_VECTOR
#define __MATRIX_VECTOR

#include <vector>
#include <stdlib.h>

class MatrixVector {

 public:
  /**
   * Multples the given A matrix by the vector B. If the sizes are mismatched,
   * reutnrs NULL. 
   */
  static vector<int> mul(vector<vector<int>> A, vector<int> b);
  
  /**
   * Creates an Size X 1 vector with each element in the matrix on the range
   * [1, max]
   */
  static vector<int> getB(int size, int max);
  
  /**
   * Creates an Size X Size matrix with each element in the matrix on the range
   * [1, max]
   */
  static vector<vector<int>> getA(int size, int max);

};






#endif
