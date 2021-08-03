#include "utils.h"

unsigned long long rdtsc(void)
{
  unsigned long long a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}

void fill_matrix_ones(double *A, int64_t dim) {
    for(int i=0;i<dim;++i) {
        A[i] = 1.0;
    }
}

void fill_matrix_zeros(double *A, int64_t dim) {
    for(int i=0;i<dim;++i) {
        A[i] = 0.0;
    }
}

void print_matrix(double *A, int64_t M, int64_t N) {
    for(int j=0;j<N;++j) {
        for(int i=0;i<M;++i) {
            printf(" %5.3f ",A[i + j*M]);
        }
        printf("\n");
    }
}

void packA(int64_t kc, double *A, int64_t incRowA, int64_t incColA, double *buffer) {
    int64_t mp = MC / MR;
    double *buffer_start = buffer;
    double *A_start = A;
    for(int k=0;k<mp;++k) {
        for(int i=0;i<kc;++i) {
            for(int j=0;j<MR;++j) {
                buffer[j] = A[j*incRowA];
            }
            A = A + incColA; // incColA == 1
            buffer = buffer + MR;
        }
        //buffer = buffer_start + MR * kc;
        A = A_start + MR * incRowA;
    }
}

void packB(int64_t kc, double *B, int64_t incRowB, int64_t incColB, double *buffer) {
    int64_t np = NC / NR;
    double *buffer_start = buffer;
    double *B_start = B;
    for(int k=0;k<np;++k) {
        for(int i=0;i<kc;++i) {
            for(int j=0;j<NR;++j) {
                buffer[j] = B[j*incColB]; // incColB == 1
            }
            B = B + incRowB;
            buffer = buffer + NR;
        }
        //buffer = buffer_start + NR * kc;
        B = B_start + NR * incColB;
    }
}

