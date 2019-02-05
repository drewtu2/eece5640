#include <iostream>
#include "table.h"

using std::cout;
using std::cin;
using std::endl;

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
    table.run(philosopher_mode);

    return 0;
}
