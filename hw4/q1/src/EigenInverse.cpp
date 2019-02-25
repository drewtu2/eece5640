#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "EigenInverse.h"

using std::cout;
using std::endl;
using std::vector;

EigenInverse::EigenInverse(vector<vector<float> > matrix, int row, int cols):
    m(row, cols) {

      for(int r = 0; r < row; ++r) {
        for(int c = 0; c < cols; ++c) {
          this->m(r, c) = matrix[r][c];
        }
      }

    }

Eigen::MatrixXf EigenInverse::run() {
  return this->m.inverse();
}
