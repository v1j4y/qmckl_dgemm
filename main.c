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
    int64_t nb = N / N;
    int64_t kb = K / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*N;

    for(int i=0;i<nb;++i) {
        nmbnb_prev = nmbnb;
        for(int k=0;k<kb;++k) {
            int64_t kc = KC;
            packB(kc, &B[k*KC*incRowB + i*N*incColB], incRowB, incColB, _B);

            idxi = i * N * incRowC;
            idxk = k * KC * incColA;

            for(int j=0;j<mb;++j) {
              //printf("[%d %d %d]===(%d %d %d)===\n",i,k,j,k,idxi,j*MC*incColC);
                packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

                //dgemm_macro_kernel(MC, KC, NC, &C[idxi + j*MC*incColC], incRowC, incColC, _A, _B);
                dgemm_macro_kernel(MC, KC, N, &C[nmbnb*MCNC], incRowC, incColC, _A, _B);
                nmbnb = nmbnb + 1;
            }
            if(k < (kb-1)) nmbnb = nmbnb_prev;
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
    double *ABlas;
    double *BBlas;
    double *DBlas;
    int64_t M, N, K;
    int64_t MBlas, NBlas, KBlas;
    int64_t incColA = 1;
    int64_t incColB = 1;
    int64_t incColC = 1;

    /*
     * We work only in factors of 16 * 14
     */

    if((MAT_DIM_M % MC) != 0){

      M = ((MAT_DIM_M/MC)+1)*MC;
    }
    else{
      M = MAT_DIM_M;
    }

    if((MAT_DIM_K % KC) != 0){
      K = ((MAT_DIM_K/KC)+1)*KC;
    }
    else{
      K = MAT_DIM_K;
    }

    if((MAT_DIM_N % NR) != 0){
      N = ((MAT_DIM_N/NR)+1)*NR;
    }
    else{
      N = MAT_DIM_N;
    }
    printf("M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",M,K,N,MC,KC,NC);

    MBlas = M;
    NBlas = N;
    KBlas = K;
    
    int64_t incRowA = K;
    int64_t incRowB = N;
    int64_t incRowC = N;

    A = (double *)malloc( M * K * sizeof(double));
    B = (double *)malloc( K * N * sizeof(double));
    C = (double *)malloc( M * N * sizeof(double));

    ABlas = (double *)malloc( MBlas * KBlas * sizeof(double));
    BBlas = (double *)malloc( KBlas * NBlas * sizeof(double));
    DBlas = (double *)malloc( MBlas * NBlas * sizeof(double));

    //fill_matrix_ones   (A, M*K);
    //fill_matrix_uniform(B, K, N);
    fill_matrix_random(A, M,K);
    fill_matrix_random(B, K, N);
    fill_matrix_zeros  (C, M*N);

    //fill_matrix_ones   (ABlas, MBlas*KBlas);
    //fill_matrix_uniform(BBlas, KBlas, NBlas);
    fill_matrix_random(ABlas, MBlas,KBlas);
    fill_matrix_random(BBlas, KBlas, NBlas);
    fill_matrix_zeros  (DBlas, MBlas*NBlas);
    //print_matrix(B,N,K);

    int64_t rep = 1;

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

    //print_matrix_ASer(C, M, N);

    const uint64_t bt0 = rdtsc();

    for(int i=0;i<rep;++i) {
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,MBlas,NBlas,KBlas,1.0,ABlas,KBlas,BBlas,NBlas,0.0,DBlas,NBlas);
    }

    const uint64_t bdt = rdtsc() - bt0;
    printf("BLAS DGEMM = %f\n", 1e-9 * bdt/rep);

    //print_matrix(DBlas, M, N);
    printf("\n-------------diff-----------------\n");
    //print_diff_matrix_AT_B(C,D, M, N);
    print_diff_matrix_ASer_B(C,DBlas, M, N);

    free(A);
    free(B);
    free(C);
    free(ABlas);
    free(BBlas);
    free(DBlas);
    return 0;
}
