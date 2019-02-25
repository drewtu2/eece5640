#ifndef __EIGEN_INVERSE
#define __EIGEN_INVERSE

#include <Eigen/Dense>
#include <vector>

using std::vector;

class EigenInverse {
  
  Eigen::MatrixXf m;

  public:
  EigenInverse(vector<vector<float> > m, int row, int cols);
  Eigen::MatrixXf run();

};



#endif
