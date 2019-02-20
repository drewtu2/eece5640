#include <iostream>
#include <unistd.h>
#include <omp.h>

#include "Philosopher.h"

using std::cout;
using std::endl;


Philosopher::Philosopher(
        int* forum, 
        vector<omp_lock_t>* forks) {

    this->thread_id = omp_get_thread_num();
    this->num_philosophers = omp_get_num_threads();
    this->forum = forum;
    this->forks = forks;
}

void* Philosopher::execute(void* args) {
    int max_eats = *(int*)args;

    int eat_count = 0;

    cout << "Philosopher " << this->thread_id << " of " << this->num_philosophers << " with max eats " << max_eats << endl;

    while(true) {
        if(eat_count > max_eats) {
            return NULL;
        }

        if(*this->forum == this->thread_id) {
            this->pickup_forks();
            cout << "Philosopher " << this->thread_id << ": Time to eat!" << endl;
            cout << "Philosopher " << this->thread_id << "holds 2 forks. All other forks are down." << endl;
            sleep(2);
            
            // Increment the person allowed to eat, reset if everyone's eaten
            *this->forum = (*this->forum + 1)%this->num_philosophers;
            cout << "Next eater is " << *this->forum << endl;

            this->putdown_forks();
            eat_count++;
        }

    }

}

void Philosopher::pickup_forks() {
    int ret = 0;
    int fork_num;

    for(int ii = 0; ii < 2; ii++) {
        fork_num = (this->thread_id + ii)%this->num_philosophers;
        
        //cout << "Philosopher " << this->thread_id << " picking up fork " << fork_num << "...";
        omp_set_lock(&forks->at(fork_num));
        if(ret != 0) {
            std::cerr << "Something went wrong picking up fork " << fork_num << endl;
        } 
        //cout << "Success!" << endl;

    }
}

void Philosopher::putdown_forks() {
    int ret = 0;
    int fork_num;

    for(int ii = 0; ii < 2; ii++) {
        fork_num = (this->thread_id + ii)%this->num_philosophers;

        //cout << "Philosopher " << this->thread_id << " putting down fork " << fork_num << "...";
        omp_unset_lock(&forks->at(fork_num));
        if(ret != 0) {
            std::cerr << "Something went wrong putting down fork " << fork_num << endl;
        } 
        //cout << "Success!" << endl;

    }
}