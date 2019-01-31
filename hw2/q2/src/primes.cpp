#include <iostream>
#include <vector>
#include <chrono>

#include "FindPrimes.h"

using std::vector;
using std::cout;
using std::cin;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;

int main() {
    int num_threads;
    int max_num;

    cout << "How many threads? :" << endl;
    cin >> num_threads;
    cout << "\nUp to? : " <<endl;
    cin >> max_num;

    auto t1 = Clock::now();
    FindPrimes finder(num_threads);
    vector<int> primes = finder.primes_to_n(max_num);
    auto t2 = Clock::now();


    cout << primes.size() << " total primes between 1 and " << max_num << endl;
    //for(vector<int>::iterator it = primes.begin(); it != primes.end(); it++) {
    //    cout << *it << endl;
    //}

    cout << "Time to find: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;

    return 0;
}
