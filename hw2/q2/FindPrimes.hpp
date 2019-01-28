#ifndef __FINDPRIMES_H
#define __FINDPRIMES_H

#include <iostream>
#include <pthread.h>
#include <vector>
#include "ParallelSieve.hpp"

using std::vector;

class FindPrimes;

/**
 * Represents a prime job
 *
 */
class PrimeJob {
  public:
    FindPrimes* instance;
    int thread_id;
    int num_threads;
    int n;
    pthread_barrier_t* b1;
    pthread_barrier_t* b2;

    /**
     * Construct a PrimeJob
     * @id: thread id
     */
    PrimeJob (FindPrimes* instance, 
            int id, 
            int num_threads,
            int n, 
            pthread_barrier_t* barrier1, 
            pthread_barrier_t* barrier2) {
        this->instance = instance;
        this->thread_id = id;
        this->num_threads = num_threads;
        this->n = n;
        this->b1 = barrier1;
        this->b2 = barrier2;
    }

};

class FindPrimes {

  private:
    vector<bool> primes;
    int k;
    int num_threads;
    pthread_t* threads;
    pthread_barrier_t b1;
    pthread_barrier_t b2;

    /**
     * Close down threads
     * @thread: the given thread
     */
    void join_threads(pthread_t thread) {
        void* ret;
        int rv = pthread_join(thread, &ret);

        PrimeJob* job = ((PrimeJob*) ret);
        delete job;
    }


    vector<int> gen_primes() {
        vector<int> ret_primes(10);
        for(vector<bool>::iterator it = primes.begin(); it != primes.end(); ++it) {
            if(*it) {
                ret_primes.push_back(*it);
            }
        }

        return ret_primes;
    }

  public:

    /**
     * Constructs a FindPrime object that is able to use a given number of 
     * threads. 
     */
    FindPrimes(int num_threads) {
        this->num_threads = num_threads;
        this->threads = new pthread_t[num_threads];
    }


    /**
     * Destructor
     */
    ~FindPrimes() {
        delete threads;
    }

    /**
     * Returns the prime numbers up to n
     * @n: the prime we're seraching for
     * @returns the nth prime
     */
    vector<int> primes_to_n(int n) {
        // Initialize the primes vector to all true
        primes = vector<bool>(n, true);
        pthread_barrier_init(&b1, NULL, n);
        pthread_barrier_init(&b2, NULL, n);
        k = 2;

        // Create and launch the parallel threads
        for(int thread_num = 0; thread_num < this->num_threads; ++thread_num) {
            std::cout << "making thread " << thread_num << " of " << this->num_threads << std::endl;
            PrimeJob job = PrimeJob(this, thread_num, num_threads, n, &b1, &b2);
            pthread_create(&threads[thread_num], NULL, &ParallelSieve::execute, &job);
        }

        // Quietly close threads
        for(int thread_num = 0; thread_num < this->num_threads; ++thread_num) {
            join_threads(threads[thread_num]);
        }

        pthread_barrier_destroy(&b1);
        pthread_barrier_destroy(&b2);

        return gen_primes();
    }

    vector<bool>* get_primes(){
        return &primes;
    }

    int get_k(){
        return k;
    }

    int set_k(int new_val){
        k = new_val;
    }
};

#endif
