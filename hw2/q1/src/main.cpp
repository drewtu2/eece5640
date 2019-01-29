#include <iostream>
#include "table.h"

using std::cout;
using std::cin;
using std::endl;

int main() {

    int num_philosophers;

    cout << "Number of philosophers: ";
    cin >> num_philosophers;

    Table table(num_philosophers);

    table.run();

    return 0;
}
