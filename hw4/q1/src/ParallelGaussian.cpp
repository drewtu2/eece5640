#include <vector>
#include <iostream>

#include "ParallelGaussian.h"

using std::endl;
using std::cout;

void print_f(float** mat, int num_row, int num_col) {
    for(int row = 0; row < num_row; ++row) {
        for(int col = 0; col < num_col; ++col) {
            cout << mat[row][col] << " ";
        }
        cout << endl;
    }
}

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

float** ParallelGaussian::run() {

    std::cout << "Start..." << endl;
    std::cout << "A: " << endl;
    print_f(this->A, 3, 3);
    std::cout << "I: " << endl;
    print_f(this->I, 3, 3);
    std::cout << endl << endl;

    gaussian_elimination();
    std::cout << "Post Elmination..." << endl;
    std::cout << "A: " << endl;
    print_f(this->A, 3, 3);
    std::cout << "I: " << endl;
    print_f(this->I, 3, 3);
    std::cout << endl << endl;


    //back_substitution();
    //std::cout << "Post Sub..." << endl;
    //std::cout << "A: " << endl;
    //print_f(this->A, 3, 3);
    //std::cout << "I: " << endl;
    //print_f(this->I, 3, 3);
    //std::cout << endl << endl;


    return I;
}

float** ParallelGaussian::get_I(int size) {
    float** I = new float*[size];
    for(int row = 0; row < size; ++row) {
        I[row] = new float[size]();
        I[row][row] = 1;
    }

    return I;
}

void ParallelGaussian::row_swap(float** A, int current_row, int num_row) {
    float *temp;
    bool found;

    temp = A[current_row];
    found = false;

    for(int row = current_row + 1; row < num_row; ++row) {
        if(A[row][row] != 0) {
            A[current_row] = A[row];
            A[row] = temp;
            found = true;
            break;
        }
    }

    if(!found) {
        cout << "does not exist" << endl;
        return;
    }

}

void ParallelGaussian::gaussian_elimination() {

    float scale;
    float factor;

    // Solve each row 1 by 1
    for (int ii = 0; ii < num_row-1; ii++) { 
        
        // Normalize the row by the first non zero element
        for (int col = 0; col < num_col; ++col) {
            this->A[ii][col] /= this->A[ii][ii];
            this->I[ii][col] /= this->I[ii][ii];
        }

        print_f(this->A, 3, 3);
        // Next row
        for (int jj = ii+ 1; jj < num_row; jj++) { 
            float ratio = this->A[jj][ii]/this->A[ii][ii]; 
            cout << ratio << endl;

            for (int kk = ii; kk < num_row; kk++) { 
                this->A[jj][kk] -= (ratio*this->A[ii][kk]); 
                this->I[jj][kk] -= (ratio*this->I[ii][kk]); 
            } 
        print_f(this->A, 3, 3);
        } 

    }
}

void ParallelGaussian::back_substitution() {
    float factor;
    for(int zeroingCol = num_col; zeroingCol >= 2; --zeroingCol) {
        for(int row = zeroingCol -1; row >= 1; --row) {
            factor = A[row][zeroingCol];

            for(int col = 1; col < num_row; ++col) {
                A[row][col] = A[row][col] - factor;
                I[row][col] = I[row][col] - factor;
            }
        }
    }
}

