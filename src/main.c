#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#include "mkl.h"
//#include "cblas.h"

#include "utils.h"
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
    double *CUnpack;
    double *_A_tile;
    double *_B_tile;
    double *_C_tile;
    double *ABlas;
    double *BBlas;
    double *DBlas;
    int64_t DIM_M, DIM_N, DIM_K;
    int64_t M, N, K;
    int64_t MBlas, NBlas, KBlas;
    int64_t incColA = 1;
    int64_t incColB = 1;
    int64_t incColC = 1;
    //srand ( time ( NULL));
    srand ( 1024);

    int64_t rep =(int64_t)atol(argv[1]);
    DIM_M =(int64_t)atol(argv[2]);
    DIM_N =(int64_t)atol(argv[3]);
    DIM_K =(int64_t)atol(argv[4]);
    printf("Reps = %ld\n",rep);
    printf("M=%ld K=%ld N=%ld \n",(long)DIM_M,(long)DIM_K,(long)DIM_N);

    qmckl_context context = qmckl_context_create();
    qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
    //init_dims_avx2_input(context, DIM_M, DIM_N, DIM_K);
    qmckl_init_pack(context, 'A', DIM_M, DIM_N, DIM_K);
    qmckl_init_pack(context, 'B', DIM_M, DIM_N, DIM_K);
    qmckl_init_pack(context, 'C', DIM_M, DIM_N, DIM_K);

    //M = qmckl_M;
    //N = qmckl_N;
    //K = qmckl_K;
    M = DIM_M;
    N = DIM_N;
    K = DIM_K;
    MBlas = M;
    NBlas = N;
    KBlas = K;
    //printf("M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)M,(long)K,(long)N,(long)ctx->MC,(long)ctx->KC,(long)ctx->NC);
    
    int64_t incRowA = K;
    int64_t incRowB = N;
    int64_t incRowC = ctx->C_tile.Nt;

    A = (double *)malloc( M * K * sizeof(double));
    B = (double *)malloc( K * N * sizeof(double));
    //_A_tile = (double *)aligned_alloc(64, MAT_DIM_M*MAT_DIM_K*2 * sizeof(double));
    //_B_tile = (double *)aligned_alloc(64, MAT_DIM_N*MAT_DIM_K*2 * sizeof(double));
    //C = (double *)malloc( M * N * sizeof(double));
    C = (double *)aligned_alloc( 64, ctx->C_tile.Mt * ctx->C_tile.Nt * sizeof(double));
    CUnpack = (double *)aligned_alloc( 64, ctx->C_tile.Mt * ctx->C_tile.Nt * sizeof(double));

    _A_tile = (double *)malloc( ctx->A_tile.Mt*ctx->A_tile.Nt * sizeof(double));
    _B_tile = (double *)malloc( ctx->B_tile.Mt*ctx->B_tile.Nt * sizeof(double));
    _C_tile = (double *)malloc( ctx->C_tile.Mt*ctx->C_tile.Nt * sizeof(double));
  
    ABlas = (double *)malloc( MBlas * KBlas * sizeof(double));
    BBlas = (double *)malloc( KBlas * NBlas * sizeof(double));
    DBlas = (double *)malloc( MBlas * NBlas * sizeof(double));

    //fill_matrix_ones   (A, M*K);
    //fill_matrix_uniform(B, K, N);
    fill_matrix_random(A, M,K);
    fill_matrix_random(B, K, N);
    fill_matrix_zeros  (C, M*N);

    copy_matrix(ABlas, A, MBlas,KBlas);
    copy_matrix(BBlas, B, KBlas,NBlas);
    //fill_matrix_ones   (ABlas, MBlas*KBlas);
    //fill_matrix_uniform(BBlas, KBlas, NBlas);
    //fill_matrix_random(ABlas, MBlas,KBlas);
    //fill_matrix_random(BBlas, KBlas, NBlas);
    fill_matrix_zeros  (DBlas, MBlas*NBlas);
    //printf("----- B     ----\n");
    //print_matrix(B,K,N);
    //printf("----- BBlas ----\n");
    //print_matrix(BBlas,K,N);
    //printf("----- A     ----\n");
    //print_matrix(A,M,K);
    //printf("----- ABlas ----\n");
    //print_matrix(ABlas,M,K);

    //int64_t rep =100000;
    int i,j=rep;

    // Tile A and B
    //tile_matrix_general(context, M, N, K, A, incRowA, incColA,
    //           B, incRowB, incColB,
    //           C, incRowC, incColC, ctx->_A_tile, ctx->_B_tile);

    qmckl_pack_matrix(context, 'A', M, K, A, incRowA);
    qmckl_pack_matrix(context, 'B', K, N, B, incRowB);
    qmckl_pack_matrix(context, 'C', M, N, C, incRowB);

    qmckl_dgemm_tiled_avx2_nn(context, A, incRowA,
               B, incRowB,
               C, incRowC);

    const uint64_t avx2t0 = rdtsc();

    for(i=0;i<j;++i) {
        qmckl_dgemm_tiled_avx2_nn(context, A, incRowA,
                   B, incRowB,
                   C, incRowC);
    }

    const uint64_t avx2dt = rdtsc() - avx2t0;
    printf("MyDGEMM(AVX2_16) = %f\n", 1e-9 * avx2dt/1);

    //print_matrix_ASer(C, qmckl_M, qmckl_N);

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
    dgemm_pack("A","T",&MB,&NB,&KB,&alpha,ABlas,&KB,ABlasp);

    // Get size of packed B
    size_t BBlasp_size = dgemm_pack_get_size("B",&MB,&NB,&KB);
    BBlasp = (double *)mkl_malloc(BBlasp_size,64);

    // Pack
    dgemm_pack("B","T",&MB,&NB,&KB,&alpha,BBlas,&NB,BBlasp);

    dgemm_compute("P","P",&MB,&NB,&KB,ABlasp,&KB,BBlasp,&NB,&beta,DBlas,&MB);

    const uint64_t bt0 = rdtsc();

    for(i=0;i<j;++i) {
        dgemm_compute("P","P",&MB,&NB,&KB,ABlasp,&KB,BBlasp,&NB,&beta,DBlas,&MB);
    }

    const uint64_t bdt = rdtsc() - bt0;
    printf("BLAS DGEMM = %f\n", 1e-9 * bdt/1);
    //print_matrix(DBlas, N, M);
    printf("\n-------------diff-----------------\n");
    //print_diff_matrix_AT_B(C,D, M, N);
    //print_diff_matrix_ASer_BT(context, C,DBlas, M, N);
    qmckl_unpack_matrix(context, CUnpack, M, N);
    print_diff_matrix_ABT(CUnpack,DBlas, M, N);
    //print_matrix(CUnpack, M, N);

    mkl_free(ABlasp);
    mkl_free(BBlasp);
    qmckl_context_destroy(context);
    free(A);
    free(B);
    free(C);
    free(CUnpack);
    free(ABlas);
    free(BBlas);
    free(DBlas);
    return 0;
}
