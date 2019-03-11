#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#define B 32
#define M B*4

double CLOCK() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC,  &t);
        return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

main(int argc, char **argv)
{
    int i,j,k,jj,kk,en;
    double start, finish, total, sum;
    int a[M][M], b[M][M], c[M][M];

    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
          a[i][j] = i+j;

    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           b[i][j] = i+j;

    for (i=0; i<M; i++)
       for (j=0; j<M; j++)
           c[i][j] = 0;

    en = B * 4;
    start = CLOCK();

    for (kk=0; kk<en; kk+=B)
      for (jj=0; jj<en; jj+=B)
         for (i=0; i< M; i++)
            for (j = jj; j< jj + B; j++) {
               sum = c[i][j];
               for (k=kk; k< kk + B; k++) {
                  sum+= a[i][j] * b[k][j];
                  }
                 c[i][j] = sum;
               }

    finish = CLOCK();
    total = finish - start;
    printf("Time for the loop = %f\n", total);

    return 0;
}
