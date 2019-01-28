#include <iostream>
#include "table.h"

using std::cout;
using std::cin;
using std::endl;

int main() {

    int num_philosphers;

    cout << "Number of philosophers: ";
    cin >> num_philosophers;

    Table table(num_philosphers);

    table.run();

    delete table;
    return 0;
}
