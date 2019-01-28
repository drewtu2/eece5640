#ifndef __PARALLEL_SIEVE
#define __PARALLEL_SIEVE

#include <iostream>
#include <vector>
#include <math.h>
#include <pthread.h>
#include "FindPrimes.hpp"

// Following Parallel Sieve from following link
// http://www.massey.ac.nz/~mjjohnso/notes/59735/seminars/01077635.pdf
#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id)+1)-BLOCK_LOW(id)) 
#define BLOCK_OWNER(index,p,n) (((p)*(index)+1)-1)/(n)

using std::vector;

class ParallelSieve {
  private:
    /**
     * Returns the next k to be used
     * @current: the current index for k
     */
    static int next_k(vector<bool>* primes, int current) {
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
    static void set_range(vector<bool>* primes, int low, int high, bool state) {
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
    static void mark_composites(vector<bool>* primes, int k, int low, int high) {
        for(int i = low; i < high; i++) {
            // if the num%k == 0, then its divisible and not prime
            if(i%k == 0) {
                primes->at(i)=false;
            }
        }
    }
  public:
    /**
     * Worker method for the nth prime function
     * @arg: PrimeJob as arguments cast as void
     */
    static void* execute(void* args) {
        PrimeJob* job = (PrimeJob*) args;
        int low = BLOCK_LOW(job->thread_id, job->num_threads, job->n);
        int high = BLOCK_HIGH(job->thread_id, job->num_threads, job->n);
        int k;
        int ret;

        std::cout << "Thread " << job->thread_id << " started." << std::endl;
        k = job->instance->get_k();
        vector<bool>* primes = job->instance->get_primes();

        while(job->n < pow(k, 2)) {
            k = job->instance->get_k();
            mark_composites(primes, k, low, high);

            pthread_barrier_wait(job->b1);

            if(job->thread_id == 0) {
                ret = next_k(primes, k);
                if(ret == -1) {
                    std::cerr << "Someting wong happend" << std::endl;
                }
                k = ret;
            }

            pthread_barrier_wait(job->b2);
        }

        delete job;
        return NULL;
    }

};

#endif
