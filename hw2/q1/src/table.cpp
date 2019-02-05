#include "table.h"
#include "Task.h"
#include "Philosopher.h"
#include "SmartPhilosopher.h"
#include "SmartPhilosopherMiddleFork.h"

Table::Table(int num_philosophers) {
    this->num_philosophers = num_philosophers;

    philosophers.resize(num_philosophers);
    forks.resize(num_philosophers);

    // Create all the forks first
    for(int i = 0; i < num_philosophers; i++) {
        pthread_mutex_init(&forks[i], NULL);
    }

    this->complete = 0;


}
Table::~Table() {

    for(int i = 0; i < num_philosophers; i++) {
        pthread_mutex_destroy(&forks[i]);
    }

    // Destroy the lock 
    pthread_mutex_destroy(&lock);
    pthread_mutex_destroy(&middle_fork);
    pthread_barrier_destroy(&barrier);

}

void* Table::thread_helper(void* args) {
    int max_eats = 2;
    Task* job = (Task*) args;
    job->execute((void*)(&max_eats));
    delete job;
}

Task* Table::philosopher_factory(
        int type, 
        int id, 
        int num, 
        int* forum, 
        vector<pthread_mutex_t>* forks) {

    if(type == NORMAL) {
        return new Philosopher(id, num, forum, forks);
    }

    if(type == SMART) {
        // intialize this barrier with all philosophers
        pthread_barrier_init(&this->barrier, NULL, num);
        pthread_mutex_init(&this->lock, NULL);
        return new SmartPhilosopher(id, num, forum, forks, &barrier, 
                &this->lock, &this->complete);
    }
    if(type == SMARTFORK) {
        // intialize this barrier with all philosophers
        pthread_barrier_init(&this->barrier, NULL, num);
        pthread_mutex_init(&this->lock, NULL);
        pthread_mutex_init(&this->middle_fork, NULL);
        return new SmartPhilosopherMiddleFork(id, num, forum, forks, &barrier, 
                &this->lock, &this->middle_fork, &this->complete);
    }
}

void Table::run(int type) {

    this->forum = 0;

    // Now begin seating the philosophers
    for(int id = 0; id < num_philosophers; id++) {
        Task* job = philosopher_factory(type, id, this->num_philosophers, 
                &this->forum, &this->forks);

        pthread_create(&philosophers[id], NULL, &thread_helper, job);
    }

    void* ret;
    // Quietly close threads
    for(int thread_num = 0; thread_num < this->num_philosophers; ++thread_num) {
        pthread_join(philosophers[thread_num], &ret);
    }
}
