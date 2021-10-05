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
    int i,j,k;

    int64_t i_tile_b, i_tile_a;

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        for(k=0;k<kb;++k) {
            int64_t kc = KC;
            packB(kc, &B[k*KC*incRowB + i*N*incColB], incRowB, incColB, _B);
    

            idxi = i * N * incRowC;
            idxk = k * KC * incColA;
    
            for(j=0;j<mb;++j) {
                packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

                dgemm_macro_kernel(MC, KC, N, &C[(i*mb + j)*MCNC], incRowC, incColC, _A, _B);
                nmbnb = nmbnb + 1;
                i_tile_a += 1;
            }
            if(k < (kb-1)) nmbnb = nmbnb_prev;
            i_tile_b += 1;
        }
    }

    return 1;
}

int dgemm_main_tiled(int64_t M, int64_t N, int64_t K, double *A, int64_t incRowA, int64_t incColA,
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
    size_t szeA = MC*KC*sizeof(double);
    size_t szeB = NC*KC*sizeof(double);
    int i,j,k,imb;
    int MCKC = MC*KC;

    int64_t i_tile_b, i_tile_a;
    double *A_tile_p __attribute__ ((aligned(64)));
    double *B_tile_p __attribute__ ((aligned(64)));
    double *C_tile_p __attribute__ ((aligned(64)));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = KC;

            idxi = i * N * incRowC;
            idxk = k * KC * incColA;

            B_tile_p = _B_tile + i_tile_b * (NC*KC);
            C_tile_p = C + (imb-1) * MCNC;
            A_tile_p = _A_tile;

            for(j=0;j<mb;++j) {
                A_tile_p += (MCKC);
                C_tile_p +=  MCNC;
                #pragma forceinline
                dgemm_macro_kernel(MC, KC, N, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
                //nmbnb = nmbnb + 1;
                i_tile_a += 1;
            }
            //if(k < (kb-1)) nmbnb = nmbnb_prev;
            i_tile_b += 1;
        }
    }
//}

    return 1;
}

int tile_matrix(int64_t M, int64_t N, int64_t K, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {

    int64_t mb = M / MC;
    int64_t nb = N / N;
    int64_t kb = K / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*N;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    for(i=0;i<nb;++i) {
        for(k=0;k<kb;++k) {
            int64_t kc = KC;
            packB(kc, &B[k*KC*incRowB + i*N*incColB], incRowB, incColB, _B);

            // Write to tiled matrix to B
            for(ii=0;ii<NC*KC;++ii) {
              _B_tile[i_tile_b * (NC*KC) + ii] = _B[ii];
            }
            i_tile_b += 1;
        }
    }

    for(k=0;k<kb;++k) {
        int64_t kc = KC;

        idxk = k * KC * incColA;

        for(j=0;j<mb;++j) {
            packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

            // Write to tiled matrix to A
            for(ii=0;ii<MC*KC;++ii) {
              _A_tile[i_tile_a * (MC*KC) + ii] = _A[ii];
            }
            i_tile_a += 1;
        }
    }

    return 1;
}

int dgemm_naive(int64_t M, int64_t N, int64_t K, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int i,j,k;
    for(i=0;i<M;++i) {
        for(j=0;j<N;++j) {
            for(k=0;k<K;++k) {
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
    //double *A_tile;
    //double *B_tile;
    //static double A_tile[MAT_DIM_M * MAT_DIM_K] __attribute__ ((aligned(64)));
    //static double B_tile[MAT_DIM_N * MAT_DIM_K] __attribute__ ((aligned(64)));
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
    printf("M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)M,(long)K,(long)N,(long)MC,(long)KC,(long)NC);

    MBlas = M;
    NBlas = N;
    KBlas = K;
    
    int64_t incRowA = K;
    int64_t incRowB = N;
    int64_t incRowC = N;

    A = (double *)malloc( M * K * sizeof(double));
    B = (double *)malloc( K * N * sizeof(double));
    C = (double *)malloc( M * N * sizeof(double));
    //A_tile = (double *)malloc( M * K * sizeof(double));
    //B_tile = (double *)malloc( K * N * sizeof(double));

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

    int64_t rep = 4;
    int i;

    // Warm up
    //for(i=0;i<rep;++i) {
    //    asm volatile ("" : : : "memory");
    //    dgemm_main(M, N, K, A, incRowA, incColA,
    //               B, incRowB, incColB,
    //               C, incRowC, incColC);
    //    //dgemm_main_tiled(M, N, K, A_tile, incRowA, incColA,
    //    //           B_tile, incRowB, incColB,
    //    //           C, incRowC, incColC);
    //}

    //fill_matrix_zeros  (C, M*N);

    // Tile A and B
    tile_matrix(M, N, K, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC, _A_tile, _B_tile);

    const uint64_t t0 = rdtsc();

    for(i=0;i<rep;++i) {
        //asm volatile ("" : : : "memory");
        //dgemm_main(M, N, K, A, incRowA, incColA,
        //           B, incRowB, incColB,
        //           C, incRowC, incColC);
        dgemm_main_tiled(M, N, K, A, incRowA, incColA,
                   B, incRowB, incColB,
                   C, incRowC, incColC);
    }

    const uint64_t dt = rdtsc() - t0;
    printf("MyDGEMM = %f\n", 1e-9 * dt/rep);

    const uint64_t bt0 = rdtsc();

    for(i=0;i<rep;++i) {
        //asm volatile ("" : : : "memory");
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,MBlas,NBlas,KBlas,1.0,ABlas,KBlas,BBlas,NBlas,0.0,DBlas,NBlas);
    }

    const uint64_t bdt = rdtsc() - bt0;
    printf("BLAS DGEMM = %f\n", 1e-9 * bdt/rep);

    //print_matrix_ASer(C, M, N);

    // Warm up
    //for(i=0;i<rep;++i) {
    //    asm volatile ("" : : : "memory");
    //    cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,MBlas,NBlas,KBlas,1.0,ABlas,KBlas,BBlas,NBlas,0.0,DBlas,NBlas);
    //}

    //print_matrix(DBlas, M, N);
    //printf("\n-------------diff-----------------\n");
    //print_diff_matrix_AT_B(C,D, M, N);
    //print_diff_matrix_ASer_B(C,DBlas, M, N);

    free(A);
    free(B);
    free(C);
    //free(A_tile);
    //free(B_tile);
    free(ABlas);
    free(BBlas);
    free(DBlas);
    return 0;
}
