#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include <Eigen/Dense>

#define EIGEN 0
#define GAUSS 1

using std::cout;
using std::endl;

class Util {
  public:

    /**
     * Prints a given matrix 
     */
    static void print_f(float** mat, int num_row, int num_col) {
        for(int row = 0; row < num_row; ++row) {
            for(int col = 0; col < num_col; ++col) {
                cout << mat[row][col] << " ";
            }
            cout << endl;
        }
    };

    /**
     * Gives a test 3x3 matrix
     */
    static float** test_a() {
        float base[9] = {1, 2, 1, 2, 1, 0, -1, 1, 2};
        //float base[9] = {5,-3, 2, -3, 2, -1, -3, 2, -2};

        float** arr = new float*[3];
        for(int row = 0; row < 3; ++row) {
            arr[row] = new float[row]();

            for(int col = 0; col < 3; ++col) {
                arr[row][col] = base[row*3 + col];
            }
        }

        return arr;
    };

    /**
     * Deletes a matrix represented by the dptr
     */
    static void delete_float_dp(float** dptr, int size) {
        for(int ii = 0; ii < size; ++ii) {
            delete[] dptr[ii];
        }
    };

    /**
     * Factory for an Inverter
     */
    static Inverter* inverter_factory(int type, float** matrix, int rows, int cols) {
        switch(type) {
            case EIGEN:
                return new EigenInverse(matrix, rows, cols);
            case GAUSS:
                return new ParallelGaussian(matrix, rows, cols);
        }
    };

    static float** to_float(const Eigen::MatrixXf& m, int rows, int cols) {
        float** arr = new float*[rows];
        for(int row = 0; row < rows; ++row) {
            arr[row] = new float[row]();

            for(int col = 0; col < cols; ++col) {
                arr[row][col] = m(row, col);
            }
        }

        return arr;

    };

    static float** random_m(int size) {
        Eigen::MatrixXf r = Eigen::MatrixXf::Random(size, size);
        r= r * 100;
        Eigen::FullPivLU<Eigen::MatrixXf> lu(r);
        while(!lu.isInvertible()) {
            r = Eigen::MatrixXf::Random(size, size);
            r = r*100;
        }

        return Util::to_float(&r, size, size);
    };

};


#endif
