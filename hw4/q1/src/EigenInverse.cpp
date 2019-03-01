#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "EigenInverse.h"

using std::cout;
using std::endl;
using std::vector;

EigenInverse::EigenInverse(float** matrix, int row, int cols):
    m(row, cols) {

      for(int r = 0; r < row; ++r) {
        for(int c = 0; c < cols; ++c) {
          this->m(r, c) = matrix[r][c];
        }
      }

    }

void EigenInverse::run() {
  this->inv = this->m.inverse();
}

float** EigenInverse::get() {
    float** arr = new float*[this->m.rows()];
    for(int row = 0; row < 3; ++row) {
        arr[row] = new float[row]();
        for(int col = 0; col < 3; ++col) {
            arr[row][col] = this->m(row, col);
        }
    }

    return arr;
}
