#ifndef __QMCKL_DGEMM_PRIVATE_H
#define __QMCKL_DGEMM_PRIVATE_H
#include <stdlib.h>
#include <stdint.h>

#if !defined(MR)
#define MR 8
#endif

#if !defined(NR)
#define NR 6
#endif

#if !defined(MR2)
#define MR2 8
#endif

#if !defined(NR2)
#define NR2 6
#endif

//typedef struct qmckl_tile_struct{
//};

//typedef struct qmckl_context_struct{
//  // Matrix dimensions
//  int64_t qmckl_M;
//  int64_t qmckl_N;
//  int64_t qmckl_K;
//
//  // Block dimensions
//  int64_t KC;
//  int64_t MC;
//  int64_t NC;
//  
//  // Container for Packed arrays
//  double* _A_tile;
//  double* _B_tile;
//  double* _C_tile;
//  
//  // Buffers for intermediates
//  double* _A;
//  double* _B;
//
//} context;

//typedef context* context_p;

typedef struct qmckl_context_struct{

int64_t qmckl_M;
int64_t qmckl_N;
int64_t qmckl_K;
int64_t MC;
int64_t NC;
int64_t KC;

double* _A_tile;
double* _B_tile;
double* _C_tile;
double* _A;
double* _B;

} qmckl_context;

typedef qmckl_context* qmckl_context_p;

void init_dims_avx512();
void init_dims_avx2();
void init_dims_avx2_input(qmckl_context_p ctxtp, int64_t DIM_M, int64_t DIM_N, int64_t DIM_K);

int dgemm_main_tiled_avx2(qmckl_context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int dgemm_main_tiled_avx2_8regs(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);


int dgemm_main_tiled_sse2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int tile_matrix(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile);

int dgemm_main_packed_avx2(qmckl_context_p ctxp, double alpha, 
			  double *A, double *B, double beta, double *C);

int dgemm_naive(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

#endif
