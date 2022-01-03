#ifndef __QMCKL_DGEMM_H
#define __QMCKL_DGEMM_H
//#include "kernel.h"
#include "kernel_avx2_16regs.h"
#include "kernel_avx2_8regs.h"
#include "kernel_sse2_8regs.h"
//#include "kernel_avx2_32regs.h"

#if !defined(MAT_DIM)
#define MAT_DIM 1024
#endif

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

#if !defined(MR1)
#define MR1 8
#endif

#if !defined(NR1)
#define NR1 2
#endif

struct context{

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

};

typedef context* context_p;

extern context ctxt;
extern context_p ctxtp;

//static double *_A_tile = NULL;
//static double *_B_tile = NULL;
//static double *_A = NULL; //[MC*KC] __attribute__ ((aligned(64)));
//static double *_B = NULL; //[NC*KC] __attribute__ ((aligned(64)));
//
//#ifdef DEFINE_QMCKL_MNK
//int64_t qmckl_M;
//int64_t qmckl_N;
//int64_t qmckl_K;
//int64_t MC;
//int64_t NC;
//int64_t KC;
//#else
//extern int64_t qmckl_M;
//extern int64_t qmckl_N;
//extern int64_t qmckl_K;
//extern int64_t MC;
//extern int64_t NC;
//extern int64_t KC;
//#endif

void init_dims_avx512();
void init_dims_avx2();
void init_dims_avx2_input(context_p ctxtp, int64_t DIM_M, int64_t DIM_N, int64_t DIM_K);

int dgemm_main_tiled(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int dgemm_main_tiled_avx2(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
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

int tile_matrix_general(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile);

int dgemm_naive(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

void unpackC(context_p ctxtp, double *B, int64_t M, int64_t N);
void free_context(context_p ctxtp);
#endif
