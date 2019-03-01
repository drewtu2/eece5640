#include "Util.h"
#include "Inverter.h"
#include "EigenInverse.h"
#include "ParallelGaussian.h"

void Util::print_f(float** mat, int num_row, int num_col) {
    for(int row = 0; row < num_row; ++row) {
        for(int col = 0; col < num_col; ++col) {
            cout << mat[row][col] << " ";
        }
        cout << endl;
    }
};

float** Util::test_a() {
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

void Util::delete_float_dp(float** dptr, int size) {
    for(int ii = 0; ii < size; ++ii) {
        delete[] dptr[ii];
    }
};

Inverter* Util::inverter_factory(int type, float** matrix, int rows, int cols) {
    cout << "\n" << endl;
    cout << "**********************************"<< endl;
    switch(type) {
        case EIGEN:
            cout << "Using EigenInverse" << endl;
            cout << "**********************************"<< endl;
            return new EigenInverse(matrix, rows, cols);
        case GAUSS:
            cout << "Using GaussianElimination" << endl;
            cout << "**********************************"<< endl;
            return new ParallelGaussian(matrix, rows, cols);
    }
};

float** Util::to_float(const Eigen::MatrixXf& m, int rows, int cols) {
    float** arr = new float*[rows];
    for(int row = 0; row < rows; ++row) {
        arr[row] = new float[cols]();

        for(int col = 0; col < cols; ++col) {
            arr[row][col] = m(row, col);
        }
    }

    return arr;

};

float** Util::random_m(int size) {
    Eigen::MatrixXf r = Eigen::MatrixXf::Random(size, size);
    r= r * 100;

    bool invt = false;

    while(!invt) {
        r = Eigen::MatrixXf::Random(size, size);
        r = r*100;
        Eigen::FullPivLU<Eigen::MatrixXf> lu(r);
        invt = lu.isInvertible();
    }

    return Util::to_float(r, size, size);
};

