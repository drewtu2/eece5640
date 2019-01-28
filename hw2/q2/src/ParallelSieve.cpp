#include <iostream>
#include <vector>
#include <math.h>
#include <pthread.h>
#include "ParallelSieve.h"

using std::vector;
using std::cout;
using std::endl;


/**
 * Construct a PrimeJob
 * @id: thread id
 */
ParallelSieve::ParallelSieve (
        int id, 
        int num_threads,
        int n, 
        vector<bool>* primes,
        int* k,
        pthread_barrier_t* barrier1, 
        pthread_barrier_t* barrier2) {
    this->thread_id = id;
    this->num_threads = num_threads;
    this->n = n;
    this->primes = primes;
    this->k = k;
    this->b1 = barrier1;
    this->b2 = barrier2;
}

/**
 * Worker method for the nth prime function
 * @arg: PrimeJob as arguments cast as void
 */
void* ParallelSieve::execute(void* args) {
    int low = BLOCK_LOW(this->thread_id, this->num_threads, this->n);
    int high = BLOCK_HIGH(this->thread_id, this->num_threads, this->n);
    int ret;

    std::cout << "Thread " << this->thread_id << " started.\n";
    std::cout << "\tlow: " << low << " high: " << high << std::endl;

    while(pow(double(*this->k), 2) < n) {
        mark_composites(*this->k, low, high);

        pthread_barrier_wait(this->b1);

        if(this->thread_id == 0) {
            ret = next_k(*this->k);
            
            if(ret == -1) {
                std::cerr << "Someting wong happend" << std::endl;
            }
            // Update the value of k
            *this->k = ret;

            cout << "next k: " << ret << endl;
        }

        pthread_barrier_wait(this->b2);
    }

    std::cout << "Thread " << this->thread_id << " finsihed." << std::endl;

    return NULL;
}

/**
 * Returns the next k to be used
 * @current: the current index for k
 */
int ParallelSieve::next_k(int current) {
    for(int i = current + 1; i < primes->size(); i++) {
        if(primes->at(i)){
            return i;
        }
    }
    return -1;
}

/**
 * Set a range from low to high to a given state. 
 *
 * @low: the low index
 * @high: the high index
 * @state: a boolean to set to
 */
void ParallelSieve::set_range(int low, int high, bool state) {
    for(int i = low; i < high; i++) {
        primes->at(i)=state;
    }
}

/**
 * Set all numbers on the range [low, high] to false if they are divisible
 * by a given k.
 * @k: the k to use
 * @low: the low index
 * @high: the high index
 */
void ParallelSieve::mark_composites(int k, int low, int high) {
    for(int i = low; i <= high; i++) {
        // if the num%k == 0, then its divisible and not prime
        if(i!= k && i%k == 0) {
            //cout << i << " is not prime!" << endl;
            primes->at(i)=false;
        }
    }
}

