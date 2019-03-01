#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include <Eigen/Dense>
#include "Inverter.h"

#define EIGEN 0
#define GAUSS 1

using std::cout;
using std::endl;

class Util {
  public:

    /**
     * Prints a given matrix 
     */
    static void print_f(float** mat, int num_row, int num_col);

    /**
     * Gives a test 3x3 matrix
     */
    static float** test_a();

    /**
     * Deletes a matrix represented by the dptr
     */
    static void delete_float_dp(float** dptr, int size);

    /**
     * Factory for an Inverter
     */
    static Inverter* inverter_factory(int type, float** matrix, int rows, int cols);

    static float** to_float(const Eigen::MatrixXf& m, int rows, int cols);

    static float** random_m(int size);

};


#endif
