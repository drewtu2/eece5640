#include <iostream>
#include <vector>
#include <chrono>

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

int main()
{
#ifdef OMP
    omp_set_num_threads(omp_get_max_threads());
#endif

    float** f = Util::test_a();
    int type = GAUSS;

    Inverter* invt = Util::inverter_factory(type, f, 3, 3);

    auto t1 = Clock::now();
    invt->run();
    auto t2 = Clock::now();

    cout << "Inverted Matrix" << endl;
    Util::print_f(invt->get(), 3, 3);

    cout << "Time to find: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;


}
