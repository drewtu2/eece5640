#include <iostream>
#include <stdlib.h>

#include "Mv.h"

using std::cout;
using std::endl;

/*
 *
 *    A B C     J  =  AJ + BK + CL
 *    D E F  *  K  =  DJ + EK + FL
 *    G H I     L  =  GJ + HK + IL
 *
 *
 */

vector<int> MatrixVector::mul(vector<vector<int>> A, vector<int> b) {
  if(A[0].size() != b.size()) {
    cout << "incompatible sizes...." << endl;
    return NULL;
  }

  // Create a vector the size of the input vector b
  vector<int> result;
  result.resize(b.size());
  int temp_sum = 0;

  for(int row = 0; row < b.size(); ++row) {
    #pragma omp parallel for reduction(+:temp_sum)
    for(int col = 0; col < b.size(); ++col) {
      temp_sum += A[row][col] * b[col];
    }
    result[b] = temp_sum;
    temp_sum = 0;
  }

  return result;
}

vector<int> MatrixVector::getB(int size, int max) {
  vector<int> result;
  result.resize(size);

  #pragma omp parallel for 
  for (int ii = 0; ii < size; ++ii) {
    result[ii] = randn() % max + 1;
  }

}

vector<vector<int>> MatrixVector::getA(int size, int max) {
  vector<vector<int>> result;
  result.resize(size);

  #pragma omp parallel for 
  for (int ii = 0; ii < size; ++ii) {
    result[ii].resize(size);
    for (int jj = 0; jj < size; ++jj) {
      result[ii][jj] = randn() % max + 1;
    }
  }

}

