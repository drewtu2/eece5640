#include <iostream>
#include <omp.h>

#include "Mv.h"

using std::cout;
using std::endl;

int main() {

    int max_threads = omp_get_max_threads();

    cout << "Max threads: "<< max_threads << endl;
    
    #pragma omp parallel 
    {
        #pragma omp critical
        {
            cout << "Hello from thread: " << omp_get_thread_num() << endl;
        }
    }

    vector<vector<int> > A {
        {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}};
    vector<int> b {
        1, 2, 3
    };

    // Create Random Matrices of correct size
    A = MatrixVector::getA(1000, 10);
    b = MatrixVector::getB(1000, 10);

    // Multiply
    vector<int> result = MatrixVector::mul(A, b);

    cout << "A * b = " << endl;;
    cout << "size(A * b)= " << result.size() << endl;

    for (std::vector<int>::const_iterator itr = result.begin(); itr != result.end(); ++itr) {
        std::cout << *itr << ' ';
    }



    return 0;
}
