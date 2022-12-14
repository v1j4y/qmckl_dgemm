#ifndef __QMCKL_DGEMM_H
#define __QMCKL_DGEMM_H
//#include "qmckl_dgemm_private.h"

typedef int32_t qmckl_exit_code;
typedef int64_t qmckl_packed_matrix;
#define QMCKL_NULL_CONTEXT (qmckl_exit_code) 0
#define QMCKL_SUCCESS (qmckl_exit_code) 0
#define QMCKL_FAILURE (qmckl_exit_code) 101

qmckl_packed_matrix qmckl_packed_matrix_create();

qmckl_exit_code qmckl_init_pack(qmckl_packed_matrix packed_matrix, unsigned char mType, int64_t M8, int64_t N8, int64_t K8);

qmckl_exit_code qmckl_pack_matrix(qmckl_packed_matrix packed_matrix, unsigned char mType, int64_t M8, int64_t N8, double *A, int64_t LDA);

qmckl_exit_code qmckl_dgemm_tiled_avx2_nn(qmckl_packed_matrix packed_matrix_A, int64_t incRowA,
                                                qmckl_packed_matrix packed_matrix_B, int64_t incRowB,
                                                qmckl_packed_matrix packed_matrix_C, int64_t incRowC);

qmckl_exit_code qmckl_dgemm_tiled(int64_t Min, int64_t Nin, int64_t Kin,
				     double *A, int64_t incRowA,
				     double *B, int64_t incRowB,
				     double *C, int64_t incRowC);

qmckl_exit_code qmckl_unpack_matrix(qmckl_packed_matrix packed_matrix, double *B, int64_t M, int64_t N);

qmckl_exit_code qmckl_packed_matrix_destroy(qmckl_packed_matrix packed_matrix);

#endif
