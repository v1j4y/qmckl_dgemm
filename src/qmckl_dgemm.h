#ifndef __QMCKL_DGEMM_H
#define __QMCKL_DGEMM_H
#include "qmckl_dgemm_private.h"

#include "kernel_avx2_16regs.h"
#include "kernel_avx2_8regs.h"
#include "kernel_sse2_8regs.h"
//#include "kernel_avx2_32regs.h"

extern qmckl_context ctxt;
extern qmckl_context_p ctxtp;

int dgemm_main_tiled(qmckl_context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int tile_matrix_general(qmckl_context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile);

void unpackC(qmckl_context_p ctxtp, double *B, int64_t M, int64_t N);
void free_context(qmckl_context_p ctxtp);
#endif
