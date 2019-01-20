// Author: Nat Tuck
// CS3650 starter code

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>

#include "barrier.h"
#include "utils.h"

barrier*
make_barrier(int nn)
{
    barrier* bb = malloc(sizeof(barrier));
    assert(bb != 0);

    pthread_mutex_init(&bb->mutex, 0);
    pthread_cond_init(&bb->cond, 0);
    bb->count = nn;  
    bb->seen  = 0;
    return bb;
}

void
barrier_wait(barrier* bb)
{
    int rv, seen;
   
    // Increment the number of people who've hit this barrier. 
    pthread_mutex_lock(&bb->mutex);
    bb->seen += 1;
    seen = bb->seen;
    pthread_mutex_unlock(&bb->mutex);

    // The last person to reach the barrier will trigger the signal. 
    if(seen >= bb->count)
    {
        rv = pthread_cond_broadcast(&bb->cond);
        check_rv(rv);
    } else
    { 
        pthread_mutex_lock(&bb->mutex);
        rv = pthread_cond_wait(&bb->cond, &bb->mutex);
        check_rv(rv);
        pthread_mutex_unlock(&bb->mutex);
    }
}


void
free_barrier(barrier* bb)
{
    free(bb);
}

