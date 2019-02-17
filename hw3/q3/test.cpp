#include <omp.h>
#include <iostream>

int main() {
    omp_set_num_threads(5);

    #pragma omp parallel 
    {

        #pragma omp single
        {
            std::cout << "num threads: " << omp_get_num_threads() << std::endl;
        }

        std::cout << "Hello world " << std::endl;
            
    }

    return 0;
}
