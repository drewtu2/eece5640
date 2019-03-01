#include <vector>
#include <iostream>

#include "ParallelGaussian.h"

using std::endl;
using std::cout;

ParallelGaussian::ParallelGaussian(float** A, int num_row, int num_col) {
    this->A = A;
    this->I = get_I(num_row);
    this->num_row = num_row;
    this->num_col = num_col;

}


ParallelGaussian::~ParallelGaussian() {
    for(int row = 0; row < num_row; ++row) {
        delete this->A[row];
        delete this->I[row];
    }
    delete this->A;
    delete this->I;

}

void ParallelGaussian::run() {
    gaussian_elimination();
    back_substitution();
}

float** ParallelGaussian::get() {
    return this->I;
}

float** ParallelGaussian::get_I(int size) {
    float** I = new float*[size];
    for(int row = 0; row < size; ++row) {
        I[row] = new float[size]();
        I[row][row] = 1;
    }

    return I;
}

void ParallelGaussian::gaussian_elimination() {

    float scale;
    float factor;
    float temp_A;
    float temp_I;

    // Solve each row 1 by 1
    for (int ii = 0; ii < num_row; ii++) { 
#pragma omp parallel
        {
            // Normalize the row by the first non zero element
#pragma omp single
            {
                temp_A = this->A[ii][ii];
            }
            //temp_I = this->I[ii][ii];
            if (temp_A != 1) {
#pragma omp for
                for (int col = 0; col < num_col; ++col) {
                    this->A[ii][col] /= temp_A;
                    this->I[ii][col] /= temp_A;
                }
            }

            if(ii != num_row - 1) {
                // Next row
#pragma omp for
                for (int jj = ii + 1; jj < num_row; jj++) { 
                    float ratio = this->A[jj][ii]/this->A[ii][ii]; 
                    for (int kk = 0; kk < num_row; kk++) { 
                        this->A[jj][kk] -= (ratio*this->A[ii][kk]); 
                        this->I[jj][kk] -= (ratio*this->I[ii][kk]); 
                    } 
                } 
            }
        }
    }
}

void ParallelGaussian::back_substitution() {
    float factor;

    // Iterate through each col
    for(int zeroingCol = num_col - 1; zeroingCol >= 1; --zeroingCol) {
        #pragma omp parallel for private(factor)
        for(int row = zeroingCol - 1; row >= 0; --row) {
            factor = A[row][zeroingCol];

            for(int col = 0; col < num_row; ++col) {
                A[row][col] = A[row][col] - factor * A[zeroingCol][col];
                I[row][col] = I[row][col] - factor * I[zeroingCol][col];
            }
        }
    }
}

