#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <sys/stat.h>
#include <float.h>

#include "float_vec.h"
#include "barrier.h"
#include "utils.h"

/*
 * Method used for spawning and joining threads is based on Prof. Tuck's lecture
 * (notes) on multithreading....
 */

/** 
 * Wrapper function to use qsort on floats vector.
 */
void
qsort_floats(floats* xs)
{
    qsort(xs->data, xs->size, sizeof(float), compare_float);
}

/**
 * Generates a vector of floats representeing the sampled input
 * @input input: the input float vector
 * @input P: the number of buckets to use
 */
floats*
sample(floats* input, int P)
{
    floats* temp_samples = make_floats(10);
    floats* samples = make_floats(10);
    seed_rng();
    int rand_index;

    // Select 3*(P-1)
    for(int ii = 0; ii<3*(P-1); ++ii)
    {
        rand_index = rand() % input->size;
        floats_push(temp_samples, input->data[rand_index]);
    }

    // Sort the sampled floats
    qsort_floats(temp_samples);
    
    // Add float minimum to samples
    floats_push(samples, FLT_MIN);

    // Iterate over every middle number in the samples and add it to the samples
    int median_index;
    for(int ii = 0; ii < (P-1); ++ii) 
    {
        median_index = (ii*3) + 1;
        floats_push(samples, temp_samples->data[median_index]);
    }
    
    // Add the maximum possible float value to the end of the samples
    floats_push(samples, FLT_MAX);

    // Free up the memory for temp samples. We're done with it. 
    free_floats(temp_samples);

    return samples;
}

void*
sort_worker(void* arg)
{
    sort_job* myJob = (sort_job*) arg;
    //free(arg);
    floats* xs = make_floats(myJob->input->size);
    float min_value = myJob->samps->data[myJob->pnum];
    float max_value = myJob->samps->data[myJob->pnum + 1];

    // Give each thread the work it needs to do here
    // Build a local array
    for(int ii = 0; ii < myJob->input->size; ++ii)
    {
        if(myJob->input->data[ii] < max_value && myJob->input->data[ii] >= min_value)
        {
            floats_push(xs, myJob->input->data[ii]);
        }
    }
    myJob->sizes[myJob->pnum] = xs->size; 
    printf("%d: start %.04f, count %ld\n", myJob->pnum, myJob->samps->data[myJob->pnum], xs->size);

    // Sort it
    qsort_floats(xs);
    
    //printf("%d reached barrier\n", myJob->pnum);
    barrier_wait(myJob->bb); // Make sure all threads have written their sample sizes
    //printf("%d crossed barrier\n", myJob->pnum);
    //if(myJob->pnum==0)
    //{
    //    printf("0: %d, 1: %d\n", (int)myJob->sizes[0], (int)myJob->sizes[1]);
    //}

    //  open(2) the output file
    int ofd = open(myJob->output, O_WRONLY);
    //printf("%d opened file with fd %d\n", myJob->pnum, ofd);

    // lseek(2) to the right spot
    // Offset is the number of bytes we should set this thread's "head" to to 
    // begin writing... 
    // It is equal to 8 bytes (inital size) + (sizeof(float) * num elements before)
    int offset;
    offset = 8 + (sum_array(myJob->sizes, myJob->pnum) * 4);
    //printf("%d offset %d\n", myJob->pnum, offset);
    lseek(ofd, offset, SEEK_SET);

    //printf("%d found seek spot\n", myJob->pnum);
    // Write your local array with write(2)
    write(ofd, xs->data, xs->size * 4);
    //printf("%d wrote to file\n", myJob->pnum);

    close(ofd);
    free_floats(xs);
    free(myJob);
    return 0;
}

pthread_t
spawn_sort_range(int pnum, floats* input, const char* output, int P, 
                floats* samps, long* sizes, barrier* bb)
{
    //Create a sort_job struct to pass the thread        
    sort_job* job = make_sort_job(pnum, input, output, P, samps, sizes, bb);
    
    //Create the actual thread to be run, give it the sortworker fx
    pthread_t thread;
    int rv = pthread_create(&thread, 0, sort_worker, job);

    //Error check and return the thread
    assert(rv == 0);
    return thread;
}

void join_sort_range(pthread_t thread)
{
    void *ret;
    int rv = pthread_join(thread, &ret);
    assert(rv == 0);

    sort_job* job = ((sort_job*) ret);
    free(job);
}

void
run_sort_workers(floats* input, const char* output, int P, 
                floats* samps, long* sizes, barrier* bb)
{
    pthread_t threads[P];

    // Spawn P threads running sort_worker
    for(int thread_num = 0; thread_num < P; ++thread_num)
    {
        threads[thread_num] = spawn_sort_range(thread_num, input, output, P, samps,  sizes, bb);
    }

    // Wait for all P threads to complete
    for(int thread_num = 0; thread_num < P; ++thread_num)
    {
        join_sort_range(threads[thread_num]);
    }
}

void
sample_sort(floats* input, const char* output, int P, long* sizes, barrier* bb)
{
    floats* samps = sample(input, P);
    run_sort_workers(input, output, P, samps, sizes, bb);
    free_floats(samps);
}

void buildInputVec(long* size, floats* inputs, const char* iname)
{
    int rv;
    //Open the input file and read the data into the input array.
    int ifd = open(iname, O_RDONLY);

    // Read the first 8 bytes of the input file. We are generating the number of
    // elements to be read into our array
    rv = read(ifd, size, 8);
    assert(rv != 0);
    float temp_element;
    for(int ii = 0; ii < *size; ++ii)
    {
        rv = read(ifd, &temp_element, 4);
        if (rv == 0)
        {
            fprintf(stderr, "Error: Read 0 bytes\n");
        }

        floats_push(inputs, temp_element);
    }
    //floats_recap(inputs, *size * 2);
    //rv = read(ifd, inputs->data, *size*sizeof(float));
    //inputs->size = *size;
    //if (rv == 0)
    //{
    //    fprintf(stderr, "Error: Read 0 bytes\n");
    //}
    close(ifd);
}

// Opens the output file and writes the number of elements to follow into it. 
// Returns the fd of the output file.  
int init_output(const char* oname, int fsize, long numElements)
{
    // Create the output file, of the same size, with ftruncate(2)
    int ofd = open(oname, O_CREAT|O_TRUNC|O_WRONLY, 0644);
    check_rv(ofd);
    ftruncate(ofd, fsize);
    
    //Write the size to the output file.
    write(ofd, &numElements, sizeof(numElements));

    return ofd;
}

int
main(int argc, char* argv[])
{
    //alarm(120);

    if (argc != 4) {
        printf("Usage:\n");
        printf("\t%s P input.dat output.dat\n", argv[0]);
        return 1;
    }

    const int P = atoi(argv[1]);
    const char* iname = argv[2];
    const char* oname = argv[3];
    int rv = 0;

    // TODO: remove this print
    //printf("Sort from %s to %s.\n", iname, oname);

    seed_rng();

    // Get some basic information of the input file. 
    struct stat st;
    rv = stat(iname, &st);
    check_rv(rv);

    const int fsize = st.st_size;   // Total size of the input file. 

    long size;                      // Number of elements in the input array;
    floats* input = make_floats(0); // The array holding our unsorted numbers
   
    // Populate the input vec with all the elements in the input file... 
    buildInputVec(&size, input, iname);
    //printf("input: %f - %f\n", input->data[0], input->data[input->size-1]);
    //printf("size: %d\n", (int)input->size);
    //printf("cap: %d\n", (int)input->cap);

    // Open and prep output file
    int ofd = init_output(oname, fsize, size);
    rv = close(ofd);
    check_rv(rv);

    barrier* bb = make_barrier(P);
    long* sizes = malloc(P * sizeof(long));

    sample_sort(input, oname, P, sizes, bb);

    free(sizes);
    free_barrier(bb);
    free_floats(input);

    return 0;
}

