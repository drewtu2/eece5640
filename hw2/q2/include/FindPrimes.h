#ifndef __FINDPRIMES_H
#define __FINDPRIMES_H

#include <iostream>
#include <pthread.h>
#include <vector>
#include "Task.h"

using std::vector;

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
    void join_threads(pthread_t thread);
    vector<int> gen_primes();

  public:

    /**
     * Constructs a FindPrime object that is able to use a given number of 
     * threads. 
     */
    FindPrimes(int num_threads);


    /**
     * Destructor
     */
    ~FindPrimes();

    /**
     * Helper function for executing tasks on threads
     */
    static void* execute_threads(void* args);

    /**
     * Returns the prime numbers up to n
     * @n: the prime we're seraching for
     * @returns the nth prime
     */
    vector<int> primes_to_n(int n);

    /**
     * Helper function for getting the list of primes
     */
    vector<bool>* get_primes();

    /**
     * Helper function for getting k
     */
    int get_k();

    /**
     * Helper function for setting k
     */
    int set_k(int new_val);
};

#endif
