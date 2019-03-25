#include <iostream>
#include <vector>
#include <cstdlib>

#include "Task.h"
#include "MethodCuda.h"

#define MAX_NUM 1000

using std::vector;
using std::cout;
using std::cin;
using std::endl;

typedef struct config {
    bool method;
    int size;
    int num_classes;
} config;

/**
 * Generates a vector of N random ints
 */
vector<int> generateNumbers(int N) {
    vector<int> numbers(N);

    for(int index = 0; index < numbers.size(); ++index) {
        numbers[index] = rand() % MAX_NUM;
    }

    return numbers;
}

Task* init() {
    int size;
    int num_classes;
    vector<int> numbers;
    srand(time(NULL));

    // Collect input on rank 0
    cout << endl << "How many numbers?: ";
    cin >>  size;
    cout << "\nNumber of classes? ";
    cin >> num_classes;
    cout << endl;
    
    numbers = generateNumbers(size);

    return new MethodCuda(numbers, num_classes);
}


int main(int argc, char *argv[]) {
    Task* histogram = init();
    histogram->run();
    histogram->print_results();

    return 0;
}
