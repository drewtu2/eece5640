#include <iostream>
#include <vector>
#include <chrono>
#define OMP

#ifdef OMP
#include <omp.h>
#endif

#include "Inverter.h"
#include "EigenInverse.h"
#include "ParallelGaussian.h"
#include "Util.h"

#define EIGEN 0
#define GAUSS 1

using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;

int main() {

    //float** f = Util::test_a();
    int type = GAUSS;
    int size;
    int num_threads;
    cout << "Size: "; std::cin >> size;
    cout << "Num threads: "; std::cin >> num_threads;

    cout << "Inversion type (Eigen: 0, Gauss: 1): "; std::cin >> type;
        cout << "\tType: " << type << endl;
        cout << "\tSize: " << size << endl;
        cout << "\tNum threads: " << num_threads << endl;
#ifdef OMP
    omp_set_num_threads(num_threads);
#endif
    
    float** f = Util::random_m(size);
    
    Inverter* invt = Util::inverter_factory(type, f, size, size);
    
    cout << "Starting Matrix" << endl;
    //Util::print_f(f, size, size);

    auto t1 = Clock::now();
    invt->run();
    auto t2 = Clock::now();

    cout << "Inverted Matrix" << endl;
    //Util::print_f(invt->get(), size, size);

    cout << "Time to find: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;


}
