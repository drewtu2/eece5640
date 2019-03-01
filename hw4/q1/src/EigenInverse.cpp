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

      lu = new Eigen::PartialPivLU<Eigen::MatrixXf>(this->m);
      cout << "Eigen num threads: " << Eigen::nbThreads( ) << endl;

    }

void EigenInverse::run() {
  //this->inv = this->m.inverse();
  this->inv = this->lu->inverse();
}

float** EigenInverse::get() {
    //std::cout << "num rows: " << this->m.rows() << " cols: " << this->m.cols() << endl;
    float** arr = new float*[this->inv.rows()];
    for(int row = 0; row < this->inv.rows(); ++row) {
        arr[row] = new float[this->inv.cols()]();
        for(int col = 0; col < this->inv.cols(); ++col) {
            arr[row][col] = this->inv(row, col);
        }
    }

    return arr;
}
