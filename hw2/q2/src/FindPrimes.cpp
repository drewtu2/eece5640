#include <iostream>
#include <pthread.h>
#include <vector>
#include "FindPrimes.h"
#include "Task.h"
#include "ParallelSieve.h"

using std::vector;


void FindPrimes::join_threads(pthread_t thread) {
    void* ret;
    int rv = pthread_join(thread, &ret);
}


vector<int> FindPrimes::gen_primes() {
    vector<int> ret_primes(0);
    int i = 0;
    for(vector<bool>::iterator it = primes.begin(); it != primes.end(); ++it) {
        if(*it == true) {
            ret_primes.push_back(i);
        }
        i++;
    }

    return ret_primes;
}

/**
 * Constructs a FindPrime object that is able to use a given number of 
 * threads. 
 */
FindPrimes::FindPrimes(int num_threads) {
    this->num_threads = num_threads;
    this->threads = new pthread_t[num_threads];
}


/**
 * Destructor
 */
FindPrimes::~FindPrimes() {
    delete threads;
}

void* FindPrimes::execute_threads(void* args) {
    Task* job = (Task*) args;
    job->execute(NULL);
}

/**
 * Returns the prime numbers up to n
 * @n: the prime we're seraching for
 * @returns the nth prime
 */
vector<int> FindPrimes::primes_to_n(int n) {
    // Initialize the primes vector to all true
    primes = vector<bool>(n, true);
    pthread_barrier_init(&b1, NULL, this->num_threads);
    pthread_barrier_init(&b2, NULL, this->num_threads);
    k = 2;

    // Create and launch the parallel threads
    for(int thread_num = 0; thread_num < this->num_threads; ++thread_num) {
        std::cout << "making thread " << thread_num << " of " << this->num_threads << std::endl;
        Task* job = new ParallelSieve(thread_num, 
                this->num_threads, 
                n, 
                &this->primes, 
                &this->k, 
                &this->b1, &this->b2);
        pthread_create(&threads[thread_num], NULL, &execute_threads, job);
    }

    // Quietly close threads
    for(int thread_num = 0; thread_num < this->num_threads; ++thread_num) {
        join_threads(threads[thread_num]);
    }

    pthread_barrier_destroy(&b1);
    pthread_barrier_destroy(&b2);

    return gen_primes();
}

vector<bool>* FindPrimes::get_primes(){
    return &primes;
}

int FindPrimes::get_k(){
    return k;
}

int FindPrimes::set_k(int new_val){
    k = new_val;
}
