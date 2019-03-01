#ifndef __EIGEN_INVERSE
#define __EIGEN_INVERSE

#include <Eigen/Dense>
#include <vector>

#include "Inverter.h"

using std::vector;

class EigenInverse: public Inverter {

  private:
    Eigen::MatrixXf m;
    Eigen::MatrixXf inv;

  public:
    EigenInverse(float** m, int row, int cols);
    void run();
    float** get();

};



#endif
