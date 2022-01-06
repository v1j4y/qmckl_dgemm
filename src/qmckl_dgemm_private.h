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

int dgemm_main_packed_avx2(context_p ctxp, double alpha, 
			  double *A, double *B, double beta, double *C);

#endif
