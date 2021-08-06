#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#include "mkl.h"
//#include "cblas.h"

#include "kernel.h"
#include "utils.h"

//static double _A[MR*KC] __attribute__ ((aligned(64)));
//static double _B[NR*KC] __attribute__ ((aligned(64)));
//static double _C[1024*4] __attribute__ ((aligned(64)));

int dgemm_main(int64_t M, int64_t N, int64_t K, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int64_t mb = M / MC;
    int64_t nb = N / NC;
    int64_t kb = K / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;

    for(int i=0;i<nb;++i) {
        for(int k=0;k<kb;++k) {
            int64_t kc = KC;
            packB(kc, &B[k*KC*incRowB + i*NC*incColB], incRowB, incColB, _B);

            idxi = i * NC * incColC;
            idxk = k * KC * incColA;

            for(int j=0;j<mb;++j) {
              //printf("===(%d %d)===\n",k,j);
                packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

                dgemm_macro_kernel(MC, KC, NC, &C[idxi + j*MC*incRowC], incRowC, incColC, _A, _B);
            }
        }
    }

    return 1;
}

int dgemm_naive(int64_t M, int64_t N, int64_t K, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    for(int i=0;i<M;++i) {
        for(int j=0;j<N;++j) {
            for(int k=0;k<K;++k) {
                C[i*N + j] = C[i*N + j] + A[i*K + k]*B[k*N + j];
            }
        }
    }

    return 1;
}


static uint64_t timer_ns() {
    const clockid_t clockid = CLOCK_MONOTONIC;
    struct timespec t;
    clock_gettime(clockid, &t);
    return 1000000000ULL * t.tv_sec + t.tv_nsec;
}

int main() {

    double *A;
    double *B;
    double *C;
    double *D;
    int64_t M, N, K;
    int64_t incColA = 1;
    int64_t incColB = 1;
    int64_t incColC = 1;

    M = MAT_DIM;
    N = MAT_DIM;
    K = MAT_DIM;
    int64_t incRowA = K;
    int64_t incRowB = N;
    int64_t incRowC = N;

    A = (double *)malloc( M * K * sizeof(double));
    B = (double *)malloc( K * N * sizeof(double));
    C = (double *)malloc( M * N * sizeof(double));
    D = (double *)malloc( M * N * sizeof(double));

    //fill_matrix_uniform(A, M,K);
    //fill_matrix_uniform(B, K, N);
    fill_matrix_random(A, M,K);
    fill_matrix_random(B, K, N);
    fill_matrix_zeros  (C, M*N);
    fill_matrix_zeros  (D, M*N);
    //print_matrix(A,M,K);

    int64_t rep = 10;

    const uint64_t t0 = rdtsc();

    for(int i=0;i<rep;++i) {
        dgemm_main(M, N, K, A, incRowA, incColA,
                   B, incRowB, incColB,
                   C, incRowC, incColC);
        //dgemm_naive(M, N, K, A, incRowA, incColA,
        //           B, incRowB, incColB,
        //           C, incRowC, incColC);
    }

    const uint64_t dt = rdtsc() - t0;
    printf("MyDGEMM = %f\n", 1e-9 * dt/rep);

    //print_matrix(C, M, N);

    const uint64_t bt0 = rdtsc();

    for(int i=0;i<rep;++i) {
        cblas_dgemm(CblasColMajor,CblasNoTrans, CblasNoTrans,M,N,K,1.0,B,M,A,K,0.0,D,M);
    }

    const uint64_t bdt = rdtsc() - bt0;
    printf("BLAS DGEMM = %f\n", 1e-9 * bdt/rep);

    //print_matrix(D, M, N);
    //printf("\n-------------diff-----------------\n");
    //print_diff_matrix(C,D, M, N);

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}
