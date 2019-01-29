#include "table.h"
#include "Task.h"
#include "Philosopher.h"

Table::Table(int num_philosophers) {
    this->num_philosophers = num_philosophers;

    philosophers.resize(num_philosophers);
    forks.resize(num_philosophers);
    
    // Create all the forks first
    for(int i = 0; i < num_philosophers; i++) {
        pthread_mutex_init(&forks[i], NULL);
    }


}

void* Table::thread_helper(void* args) {
    int max_eats = 2;
    Task* job = (Task*) args;
    job->execute((void*)(&max_eats));
    delete job;
}

void Table::run() {

    this->forum = 0;

    // Now begin seating the philosophers
    for(int id = 0; id < num_philosophers; id++) {
        Task* job = new Philosopher(id, this->num_philosophers, 
                &this->forum, &this->forks);

        pthread_create(&philosophers[id], NULL, &thread_helper, job);
    }
    
    void* ret;
    // Quietly close threads
    for(int thread_num = 0; thread_num < this->num_philosophers; ++thread_num) {
        pthread_join(philosophers[thread_num], &ret);
    }

    for(int i = 0; i < num_philosophers; i++) {
        pthread_mutex_destroy(&forks[i]);
    }

}
