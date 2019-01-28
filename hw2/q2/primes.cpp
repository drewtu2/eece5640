#include <iostream>
#include <vector>
#include <pthread.h>
#include <math.h>


// Following Parallel Sieve from following link
// http://www.massey.ac.nz/~mjjohnso/notes/59735/seminars/01077635.pdf
#define BLOCK_LOW(id,p,n) ((i)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id)+1)-BLOCK_LOW(id)) 
#define BLOCK_OWNER(index,p,n) (((p)*(index)+1)-1)/(n)

using std::vector;


typedef struct PrimeJob {
  int thread_id;
  int n;
} PrimeJob;

class FindPrimes {

 private:
  vector<bool> primes;
  int k;
  int num_threads;
  pthread_t* threads;
  pthread_barrier_t* b1;
  pthread_barrier_t* b2;


  /**
   * Worker method for the nth prime function
   * @arg: PrimeJob as arguments cast as void
   */
  void* parallel_sieve(void* arg) {
    PrimeJob job = (PrimeJob*) arg;
    int low = BLOCK_LOW(job->thread_id, this.num_threads, job->n);
    int high = BLOCK_HIGH(job->thread_id, this.num_threads, job->n);
    int multiple;
    int ret;
    //this.set_range(low, high, true);
    
    while(n < pow(k, 2)) {
      multiple = this.k;
      this.mark_composites(multiple, low, high);

      pthread_barrier_wait(this.b1);

      if(job->thread_id == 0) {
        ret = this.next_k(multiple);
        if(ret == -1) {
          std::cerr << "Someting wong happend" << std::endl;
        }
        this.k = ret;
      }
      
      pthread_barrier_wait(this.b2);
    }

    free(job);
    return 0;
  }

  /**
   * Returns the next k to be used
   * @current: the current index for k
   */
  int next_k(int current) {
    for(int i = current + 1; i < primes.length(); i++) {
      if(primes(i)){
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
  void set_range(int low, int high, bool state) {
    for(int i = low; i < high; i++) {
      primes.at(i)=state;
    }
  }

  /**
   * Set all numbers on the range [low, high] to false if they are divisible
   * by a given k.
   * @k: the k to use
   * @low: the low index
   * @high: the high index
   */
  void mark_composites(int k, int low, int high) {
    for(int i = low; i < high; i++) {
      // if the num%k == 0, then its divisible and not prime
      if(i%k == 0) {
        primes.at(i)=false;
      }
    }
  }

  /**
   * Close down threads
   * @thread: the given thread
   */
  void join_sort_range(pthread_t thread) {
    void* ret;
    int rv = pthread_join(thread, &ret);
    
    PrimeJob* job = ((PrimeJob*) ret);
    delete job;
  }

  /**
   * Construct a PrimeJob
   * @id: thread id
   */
  PrimeJob make_PrimeJob(int id, int n) {
    PrimeJob* job = new PrimeJob;
    job->thread_id = id;
    job->n = n;

    return job;
  }

  vector<int> get_primes() {
    vector<int> ret_primes();
    for(vector<bool>::iterator it = primes.begin(); it != primes.end(); ++it) {
      if(*it) {
        ret_primes.append(*it);
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
    this.head = 0;
    this.num_threads = num_threads;
    this.threads = new pthread_t[num_threads];
  }

  /**
   * Destructor
   */
  ~FindPrimes() {
    delete this.threads;
  }

  /**
   * Returns the prime numbers up to n
   * @n: the prime we're seraching for
   * @returns the nth prime
   */
  vector<int> primes_to_n(int n) {
    // Initialize the primes vector to all true
    this.primes = vector<bool>(n, true);
    ret = pthread_barrier_init(this.b1, NULL, n);
    assert(ret == 0);
    ret = pthread_barrier_init(this.b2, NULL, n);
    assert(ret == 0);
    this.k = 2;
      
    // Create and launch the parallel threads
    for(int thread_num = 0; thread_num < num_threads; ++thread_num) {
      PrimeJob* job = make_PrimeJob(thread_num);
      this.threads[thread_num] = pthread_create(thread_num, NULL, &this.parallel_sieve, job);
    }
    
    // Quietly close threads
    for(int thread_num = 0; thread_num < this.num_threads; ++thread_num) {
      join_threads(this.threads[thread_num]);
    }

    pthread_barrier_destroy(&b1);
    pthread_barrier_destroy(&b2);

    return this.get_primes();
  }
};
