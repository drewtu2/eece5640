#include <iostream>
#include <stdio.h>
#include <cstdlib>
#define n 64

using std::cout;
using std::endl;


namespace naive {
    extern void launch_naive_stencil(float* input_a, float* input_b, int side_size);
}
namespace shared {
    extern void launch_shared_stencil(float* input_a, float* input_b, int side_size);
}

void convert_b(float* input, float b[][n][n], int side_size) {
    int count = 0;
    
    for(int ii = 0; ii < n; ++ii) {
        for(int jj = 0; jj < n; ++jj) {
            for(int kk = 0; kk < n; ++kk) {
                input[count] = b[ii][jj][kk];
                count++;
            }   
        }   
    }
}

void set_b(float b[][n][n]) {
    int count = 0;
    
    for(int ii = 0; ii < n; ++ii) {
        for(int jj = 0; jj < n; ++jj) {
            for(int kk = 0; kk < n; ++kk) {
                b[ii][jj][kk] = count%100 ;
                //b[ii][jj][kk] = rand()%100 ;
                count++;
            }   
        }   
    }

}


void cpu_stencil(float a[n][n][n], float b[n][n][n]) {
// Access is currently zyx... 
    for (int i=1; i<n-1; i++) {
        for (int j=1; j<n-1; j++) {
            for (int k=1; k<n-1; k++) {
                a[i][j][k]=0.8*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k] +
                        b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]);
            }
        }
    }

}

void compare(float cpu_a[][n][n], float* gpu_a, bool a) {
    char t = 'b';
    if (a) {
        t = 'a';
    }   

    int diff_count = 0;
    int same_count = 0;
    int count = 0;
    for(int ii = 0; ii < n; ++ii) {
        for(int jj = 0; jj < n; ++jj) {
            for(int kk = 0; kk < n; ++kk) {
                if(cpu_a[ii][jj][kk] != gpu_a[count]) {
                    //printf("cpu_%c: %E \t gpu_%c: %E: \t count: %d\n", 
                    //    t, cpu_a[ii][jj][kk], t, gpu_a[count], count);
                    diff_count++;
                } else {
                    same_count++;
                }
                count++;
            }   
        }   
    }

    if(diff_count==0) {
        printf("Whoopee everything worked!\n");
    } else {
        printf("%d errors...\n", diff_count);
        printf("%d correct...\n", same_count);
    }
}

int main() {
    
    srand(0);

    int serialized_size = n*n*n;

    float a[serialized_size];
    float b[serialized_size];

    float cpu_a[n][n][n], cpu_b[n][n][n];

    set_b(cpu_b);
    convert_b(b, cpu_b, n);
    
    compare(cpu_b, b, false);
    
    cpu_stencil(cpu_a, cpu_b);
    cout << "a[10][10][5]: " << cpu_a[10][10][5] << endl;
    
    naive::launch_naive_stencil(a, b, n);

    compare(cpu_a, a, true);
    
    shared::launch_shared_stencil(a, b, n);
    //
    compare(cpu_a, a, true);

    return 0;
}
