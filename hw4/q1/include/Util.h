#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#define EIGEN 0
#define GAUSS 1

using std::cout;
using std::endl;

class Util {
  public:
    static void print_f(float** mat, int num_row, int num_col) {
        for(int row = 0; row < num_row; ++row) {
            for(int col = 0; col < num_col; ++col) {
                cout << mat[row][col] << " ";
            }
            cout << endl;
        }
    }

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
    }

    static void delete_float_dp(float** dptr, int size) {
        for(int ii = 0; ii < size; ++ii) {
            delete[] dptr[ii];
        }
    }
    static Inverter* inverter_factory(int type, float** matrix, int rows, int cols) {
        switch(type) {
            case EIGEN:
                return new EigenInverse(matrix, rows, cols);
            case GAUSS:
                return new ParallelGaussian(matrix, rows, cols);
        }
    }



};


#endif
