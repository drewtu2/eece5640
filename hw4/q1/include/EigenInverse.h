#ifndef __EIGEN_INVERSE
#define __EIGEN_INVERSE

#include <Eigen/Dense>
#include <vector>

#include "Inverter.h"

using std::vector;

class EigenInverse: public Inverter {

  private:
    Eigen::MatrixXf m;
    Eigen::MatrixXf i;
    Eigen::MatrixXf inv;
    Eigen::PartialPivLU<Eigen::MatrixXf>* lu;

  public:
    EigenInverse(float** m, int row, int cols);
    void run();
    float** get();

};



#endif
