#include <iostream>
#include <unistd.h>
#include <omp.h>

#include "SmartPhilosopher.h"

using std::cout;
using std::endl;


SmartPhilosopher::SmartPhilosopher(
        int* forum, 
        vector<omp_lock_t>* forks,
        int* finish_count) {

    this->thread_id = omp_get_thread_num();
    this->num_philosophers = omp_get_num_threads();

    this->forum = forum;
    this->forks = forks;
    this->finish_count = finish_count;
}

void* SmartPhilosopher::execute(void* args) {
    int max_eats = *(int*)args;

    int eat_count = 0;
    int fin = false;

    while(!fin) {
        if(this->check_finished(eat_count, max_eats)) {
        #pragma omp critical 
        {
            if(*this->finish_count == this->num_philosophers)
            {
                cout << "thread: " << this->thread_id << "finish count: " 
                    << *this->finish_count<< endl;
                fin = true;
            }
        }
        } else {
            // Only do this if we're eating... otherwise go straight to waiting.
            if(this->time_to_eat()) {
                this->pickup_forks();
                sleep(1);
                this->putdown_forks();
                eat_count++;

                if(this->check_finished(eat_count, max_eats)) {
                #pragma omp critical
                {
                    *this->finish_count = *this->finish_count + 1;
                }
                }

            }

        }

        // Synchronize all philosphers. Can't update group until all have finished
        #pragma omp barrier
        this->print_update();
        // We've update the group in thread 0, now continue
        #pragma omp barrier
    }
    return NULL;

}

bool SmartPhilosopher::check_finished(int eat_count, int max_eats) {
    return eat_count > max_eats;
}

void SmartPhilosopher::print_update() {
    // Update the which group can eat
    // g0: 0, 2, 4, ...
    // g1: 1, 3, 5, ...
    // g2: n-1
    if(this->thread_id == 0) {
        cout << "Philosopher Group " << *this->forum << ": Time to eat!" << endl;
        if(*this->forum == 0) {
            cout << "Even philosophers eating... all forks used" << endl;
        } else if(*this->forum == 1) {
            cout << "Odd philosophers eating... all forks used" << endl;
        } else if(*this->forum == 2) {
            cout << "Last philosopher eating... only two forks used" << endl;
        }

        *this->forum = (*this->forum + 1)%3;
    }

}

bool SmartPhilosopher::time_to_eat() {
    // This is the last philosopher
    // eats with group 2
    if (this->thread_id == this->num_philosophers - 1) {
        return *this->forum == 2;
    }

    // This represents phil 0, 2, 4, ...
    // eats with group 0
    if (this->thread_id % 2 == 0) {
        return *this->forum == 0;
    }

    // This represents phil 1, 3, 5, ...
    // eats with group 1
    if (this->thread_id % 2 == 1) {
        return *this->forum == 1;
    }
}

void SmartPhilosopher::pickup_forks() {
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

void SmartPhilosopher::putdown_forks() {
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
