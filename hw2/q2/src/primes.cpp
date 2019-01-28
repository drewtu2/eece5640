#include <iostream>
#include <vector>
#include "FindPrimes.h"

using std::vector;
using std::cout;
using std::cin;
using std::endl;

int main() {
    int num_threads;
    int max_num;

    cout << "How many threads? :" << endl;
    cin >> num_threads;
    cout << "\nUp to? : " <<endl;
    cin >> max_num;

    FindPrimes finder(num_threads);
    
    vector<int> primes = finder.primes_to_n(max_num);

    cout << primes.size() << " total primes between 1 and " << max_num << endl;
    for(vector<int>::iterator it = primes.begin(); it != primes.end(); it++) {
        cout << *it << endl;
    }

    return 0;
}
