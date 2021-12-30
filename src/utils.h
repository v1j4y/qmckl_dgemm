#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//#include "mipp.h"

#if !defined(MAT_DIM)
#define MAT_DIM 1024
#endif

//#if !defined(MAT_DIM_M)
//#define MAT_DIM_M MAT_DIM
//#endif
//
//#if !defined(MAT_DIM_K)
//#define MAT_DIM_K MAT_DIM
//#endif
//
//#if !defined(MAT_DIM_N)
//#define MAT_DIM_N MAT_DIM
//#endif

//#if !defined(MC)
//#if MAT_DIM_M <= 192
//#define MC 192
//#endif
//#endif
//
//#if !defined(NC)
//#if MAT_DIM_N <= 192
//#define NC 192
//#endif
//#endif

//#if !defined(KC)
//#define KC 512
//#endif

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

//static double _A[MC*KC] __attribute__ ((aligned(64)));
//static double _B[NC*KC] __attribute__ ((aligned(64)));

//static double _A_tile[MAT_DIM_M*MAT_DIM_K] __attribute__ ((aligned(64)));
//static double _A_tile[MAT_DIM_M*MAT_DIM_K*2] __attribute__ ((aligned(64)));
//static double _B_tile[MAT_DIM_N*MAT_DIM_K*2] __attribute__ ((aligned(64)));

static int  idxlist[64] = {
      0 + 0, 8 + 1, 16 + 2, 24 + 3, 32 + 4, 40 + 5, 48 + 6, 54 + 7,
      8 + 0, 0 + 1, 24 + 2, 16 + 3, 40 + 4, 32 + 5, 54 + 6, 48 + 7,
      16 + 0, 24 + 1, 0 + 2, 8 + 3, 48 + 4, 54 + 5, 32 + 6, 40 + 7,
      24 + 0, 16 + 1, 8 + 2, 0 + 3, 54 + 4, 48 + 5, 40 + 6, 32 + 7,
      32 + 0, 40 + 1, 48 + 2, 54 + 3, 0 + 4, 8 + 5, 16 + 6, 24 + 7,
      40 + 0, 32 + 1, 54 + 2, 48 + 3, 8 + 4, 0 + 5, 24 + 6, 16 + 7,
      48 + 0, 54 + 1, 32 + 2, 40 + 3, 16 + 4, 24 + 5, 0 + 6, 8 + 7,
      54 + 0, 48 + 1, 40 + 2, 32 + 3, 24 + 4, 16 + 5, 8 + 6, 0 + 7
                       };



unsigned long long rdtsc(void);
void fill_matrix_random(double *dA, int64_t dM, int64_t dN);
void fill_matrix_ones(double *A, int64_t dim);
void fill_matrix_uniform(double *A, int64_t M, int64_t N);
void fill_matrix_zeros(double *A, int64_t dim);
void print_matrix(double *A, int64_t M, int64_t N);
void print_diff_matrix(double *A, double *B, int64_t M, int64_t N);
void print_diff_matrix_AT_B(double *A, double *B, int64_t M, int64_t N);
void packA(int64_t kc, double *A, int64_t incRowA, int64_t incColA, double *buffer);
void packB(int64_t kc, double *B, int64_t incRowB, int64_t incColB, double *buffer);
void packA_general(int64_t kc, int64_t MCmax, double *A, int64_t dimRowA, int64_t dimColA, double *buffer);
void packB_general(int64_t kc, int64_t NCmax, double *B, int64_t dimRowB, int64_t dimColB, double *buffer);
void print_diff_matrix_ASer_B(double *A, double *B, int64_t M, int64_t N);
void print_diff_matrix_ASer_BT(double *A, double *B, int64_t M, int64_t N);
void print_matrix_ASer(double *A, int64_t M, int64_t N);
#endif 
