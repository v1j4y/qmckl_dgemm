#ifndef __QMCKL_DGEMM_H
#define __QMCKL_DGEMM_H
#include "qmckl_dgemm_private.h"

typedef int64_t qmckl_context;
typedef int32_t qmckl_exit_code;
#define QMCKL_NULL_CONTEXT (qmckl_context) 0
#define QMCKL_SUCCESS (qmckl_context) 0
#define QMCKL_FAILURE (qmckl_context) 101

qmckl_context qmckl_context_create();

qmckl_context qmckl_tile_matrix_create();

qmckl_exit_code qmckl_init_pack(qmckl_context context, qmckl_tile_matrix tile_matrix, unsigned char mType, int64_t M8, int64_t N8, int64_t K8);

qmckl_exit_code qmckl_pack_matrix(qmckl_context context, unsigned char mType, int64_t M8, int64_t N8, double *A, int64_t LDA);

qmckl_exit_code qmckl_dgemm_tiled_avx2_nn(qmckl_context context, double *A, int64_t incRowA,
                                                double *B, int64_t incRowB,
                                                double *C, int64_t incRowC);

qmckl_exit_code qmckl_dgemm_tiled_NN(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin,
				     double *A, int64_t incRowA,
				     double *B, int64_t incRowB,
				     double *C, int64_t incRowC);

qmckl_exit_code qmckl_unpack_matrix(qmckl_context context, double *B, int64_t M, int64_t N);

qmckl_exit_code qmckl_context_destroy(qmckl_context context);
#endif
