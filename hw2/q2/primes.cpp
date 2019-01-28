#include <iostream>
#include <vector>
#include <pthread.h>
#include <math.h>
#include "FindPrimes.hpp"


int main() {
    FindPrimes finder(2);

    finder.primes_to_n(10);

    return 0;
}
