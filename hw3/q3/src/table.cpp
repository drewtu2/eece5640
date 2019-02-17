#include <omp.h>
#include <iostream>

#include "table.h"
#include "Task.h"
#include "Philosopher.h"
#include "SmartPhilosopher.h"

Table::Table(int num_philosophers) {
    this->num_philosophers = num_philosophers;

    forks.resize(num_philosophers);

    // Create all the forks first
    for(int i = 0; i < num_philosophers; i++) {
        omp_init_lock(&forks[i]);
    }

    this->complete = 0;
}

Table::~Table() {

    for(int i = 0; i < this->num_philosophers; i++) {
        omp_destroy_lock(&forks[i]);
    }
}

Task* Table::philosopher_factory(
        int type, 
        int* forum, 
        vector<omp_lock_t>* forks) {

    if(type == NORMAL) {
        return new Philosopher(forum, forks);
    }

    if(type == SMART) {
        // intialize this barrier with all philosophers
        return new SmartPhilosopher(forum, forks, &this->complete);
    }
}

void Table::run(int type) {

    this->forum = 0;
    int max_eats = 2;
    
    omp_set_num_threads(5);

    std::cout << "Setting up with " << this->num_philosophers << std::endl;
    #pragma omp parallel 
    {
        Task* job = philosopher_factory(type, &this->forum, &this->forks);
        job->execute((void *)&max_eats);
    }
}
