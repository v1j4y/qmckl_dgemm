#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#include "mkl.h"
//#include "cblas.h"

#include "utils.h"
#define DEFINE_QMCKL_MNK
#include "qmckl_dgemm.h"

// DIRECT JIT MKL
void *jitter;


static uint64_t timer_ns() {
    const clockid_t clockid = CLOCK_MONOTONIC;
    struct timespec t;
    clock_gettime(clockid, &t);
    return 1000000000ULL * t.tv_sec + t.tv_nsec;
}

int main(int argc, char *argv[]) {

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

    init_dims_avx2();

    //M = qmckl_M;
    //N = qmckl_N;
    //K = qmckl_K;
    M = MAT_DIM_M;
    N = MAT_DIM_N;
    K = MAT_DIM_K;
    MBlas = M;
    NBlas = N;
    KBlas = K;
    printf("M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)M,(long)K,(long)N,(long)MC,(long)KC,(long)NC);
    
    int64_t incRowA = K;
    int64_t incRowB = N;
    int64_t incRowC = qmckl_N;

    A = (double *)malloc( M * K * sizeof(double));
    B = (double *)malloc( K * N * sizeof(double));
    //_A_tile = (double *)aligned_alloc(64, MAT_DIM_M*MAT_DIM_K*2 * sizeof(double));
    //_B_tile = (double *)aligned_alloc(64, MAT_DIM_N*MAT_DIM_K*2 * sizeof(double));
    //C = (double *)malloc( M * N * sizeof(double));
    C = (double *)aligned_alloc( 64, qmckl_M * qmckl_N * sizeof(double));

    ABlas = (double *)malloc( MBlas * KBlas * sizeof(double));
    BBlas = (double *)malloc( KBlas * NBlas * sizeof(double));
    DBlas = (double *)malloc( MBlas * NBlas * sizeof(double));

    fill_matrix_ones   (A, M*K);
    fill_matrix_uniform(B, K, N);
    //fill_matrix_random(A, M,K);
    //fill_matrix_random(B, K, N);
    fill_matrix_zeros  (C, M*N);

    fill_matrix_ones   (ABlas, MBlas*KBlas);
    fill_matrix_uniform(BBlas, KBlas, NBlas);
    //fill_matrix_random(ABlas, MBlas,KBlas);
    //fill_matrix_random(BBlas, KBlas, NBlas);
    fill_matrix_zeros  (DBlas, MBlas*NBlas);
    //print_matrix(B,N,K);

    int64_t rep =(int64_t)atol(argv[1]);
    printf("Reps = %ld\n",rep);
    //int64_t rep =100000;
    int i,j=rep;

    // Tile A and B
    tile_matrix_general(M, N, K, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC, _A_tile, _B_tile);

    //dgemm_main_tiled(M, N, K, A, incRowA, incColA,
    //           B, incRowB, incColB,
    //           C, incRowC, incColC);

    //const uint64_t t0 = rdtsc();

    //for(i=0;i<j;++i) {
    //    dgemm_main_tiled(M, N, K, A, incRowA, incColA,
    //               B, incRowB, incColB,
    //               C, incRowC, incColC);
    //}

    //const uint64_t dt = rdtsc() - t0;
    //printf("MyDGEMM(AVX512) = %f\n", 1e-9 * dt/1);

    dgemm_main_tiled_avx2(qmckl_M, qmckl_N, qmckl_K, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC);

    const uint64_t avx2t0 = rdtsc();

    for(i=0;i<j;++i) {
        dgemm_main_tiled_avx2(qmckl_M, qmckl_N, qmckl_K, A, incRowA, incColA,
                   B, incRowB, incColB,
                   C, incRowC, incColC);
    }

    const uint64_t avx2dt = rdtsc() - avx2t0;
    printf("MyDGEMM(AVX2_16) = %f\n", 1e-9 * avx2dt/1);

    //print_matrix_ASer(C, qmckl_M, qmckl_N);

    //dgemm_main_tiled_avx2_8regs(M, N, K, A, incRowA, incColA,
    //           B, incRowB, incColB,
    //           C, incRowC, incColC);

    //const uint64_t sse2t0 = rdtsc();

    //for(i=0;i<j;++i) {
    //    dgemm_main_tiled_avx2_8regs(M, N, K, A, incRowA, incColA,
    //               B, incRowB, incColB,
    //               C, incRowC, incColC);
    //}

    //const uint64_t sse2dt = rdtsc() - sse2t0;
    //printf("MyDGEMM(AVX2_8) = %f\n", 1e-9 * sse2dt/1);

    //dgemm_main_tiled_sse2(M, N, K, A, incRowA, incColA,
    //           B, incRowB, incColB,
    //           C, incRowC, incColC);

    //const uint64_t sse2t0 = rdtsc();

    //for(i=0;i<j;++i) {
    //    dgemm_main_tiled_sse2(M, N, K, A, incRowA, incColA,
    //               B, incRowB, incColB,
    //               C, incRowC, incColC);
    //}

    //const uint64_t sse2dt = rdtsc() - sse2t0;
    //printf("MyDGEMM(AVX2_16) = %f\n", 1e-9 * sse2dt/1);

    // MKL
    //mkl_jit_status_t status = mkl_jit_create_dgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, MBlas, NBlas, KBlas, 1.0, KBlas, NBlas, 0.0, NBlas);
    //if(MKL_JIT_ERROR == status){
    //  printf("Error in MKL\n");
    //  return(1);
    //}
    //dgemm_jit_kernel_t mkl_dgemm = mkl_jit_get_dgemm_ptr(jitter);

    //mkl_dgemm(jitter,ABlas,BBlas,DBlas);

    const int MB = MBlas;
    const int NB = NBlas;
    const int KB = KBlas;
    const double alpha=1.0;
    const double beta=0.0;
    double *ABlasp;
    double *BBlasp;

    // Get size of packed A
    size_t ABlasp_size = dgemm_pack_get_size("A",&MB,&NB,&KB);
    ABlasp = (double *)mkl_malloc(ABlasp_size,64);

    // Pack
    dgemm_pack("A","N",&MB,&NB,&KB,&alpha,ABlas,&MB,ABlasp);

    // Get size of packed B
    size_t BBlasp_size = dgemm_pack_get_size("B",&MB,&NB,&KB);
    BBlasp = (double *)mkl_malloc(BBlasp_size,64);

    // Pack
    dgemm_pack("B","T",&MB,&NB,&KB,&alpha,BBlas,&NB,BBlasp);

    //cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,MBlas,NBlas,KBlas,1.0,ABlas,KBlas,BBlas,NBlas,0.0,DBlas,NBlas);
    dgemm_compute("P","P",&MB,&NB,&KB,ABlasp,&KB,BBlasp,&NB,&beta,DBlas,&MB);

    const uint64_t bt0 = rdtsc();

    for(i=0;i<j;++i) {
        dgemm_compute("P","P",&MB,&NB,&KB,ABlasp,&KB,BBlasp,&NB,&beta,DBlas,&MB);
    }

    const uint64_t bdt = rdtsc() - bt0;
    printf("BLAS DGEMM = %f\n", 1e-9 * bdt/1);
    //print_matrix(DBlas, M, N);
    //printf("\n-------------diff-----------------\n");
    //print_diff_matrix_AT_B(C,D, M, N);
    //print_diff_matrix_ASer_BT(C,DBlas, M, N);

    mkl_free(ABlasp);
    mkl_free(BBlasp);
    free(A);
    free(B);
    free(C);
    free(ABlas);
    free(BBlas);
    free(DBlas);
    return 0;
}
