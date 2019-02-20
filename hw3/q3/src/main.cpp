#include <iostream>
#include <omp.h>
#include <chrono>

#include "table.h"

using std::cout;
using std::cin;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;


int main() {

    int num_philosophers;
    int philosopher_mode;

    cout << "Number of philosophers: ";
    cin >> num_philosophers;
    cout << "Philsopher mode: " << endl;
    cout << "\t0: Naive (priority through serialization)" << endl;
    cout << "\t1: Smart (alternating 3 groups)" << endl;
    cout << "\t2: Smart w/ Middle Fork (alternating 2 groups)" << endl;
    cin >> philosopher_mode;

    if(philosopher_mode > 2 || philosopher_mode < 0) {
        cout << "invalid mode...";
        return 0;
    }

    // Actually do stufff
    Table table(num_philosophers);

    auto t1 = Clock::now();
    table.run(philosopher_mode);
    auto t2 = Clock::now();
    cout << "Time to find: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms" << std::endl;


    return 0;
}
