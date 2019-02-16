#include "CheckDivisible.h"
#include "TaskChecker.h"

CheckDivisible::CheckDivisible() {
    this->num_threads = 4;
    this->max = 1000;
    this->divisors.push_back(3);
    this->divisors.push_back(4);

    threads.resize(this->num_threads);

}

CheckDivisible::CheckDivisible(int num_threads, int max, vector<int> divisors) {
    this->num_threads = num_threads;
    this->max = max;
    this->divisors = divisors;

    threads.resize(num_threads);

}

CheckDivisible::~CheckDivisible() {

}

void* CheckDivisible::thread_helper(void* args) {
    Task* job = (Task*) args;
    job->execute(NULL);
    delete job;
}

void CheckDivisible::run() {

    int lb, ub;
    int block_size = this->max/num_threads;

    // Now begin seating the philosophers
    for(int id = 0; id < this->num_threads; id++) {
        lb = block_size * id;
        ub = block_size * (id + 1);

        Task* job = new TaskChecker(id, lb, ub, divisors);

        pthread_create(&this->threads[id], NULL, &CheckDivisible::thread_helper, job);
    }

    void* ret;
    // Quietly close threads
    for(int thread_num = 0; thread_num < this->num_threads; ++thread_num) {
        pthread_join(threads[thread_num], &ret);
    }
}
