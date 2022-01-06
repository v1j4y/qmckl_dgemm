#ifndef __QMCKL_DGEMM_PRIVATE_H
#define __QMCKL_DGEMM_PRIVATE_H
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

typedef int64_t qmckl_context;
typedef int32_t qmckl_exit_code;

typedef struct qmckl_tile_struct{
  // Container for Packed arrays
  double* data;
  // Type of Packing (A or B or C)
  char mType;
  // Matrix dimensions
  int64_t Mt;
  int64_t Nt;
  // Block dimensions
  int64_t MCt;
  int64_t NCt;
  // Tile dimensions
  int64_t MRt;
  int64_t NRt;
} qmckl_tile_struct;

typedef struct qmckl_context_struct{

  //int64_t qmckl_M;
  //int64_t qmckl_N;
  //int64_t qmckl_K;
int64_t MC;
int64_t NC;
int64_t KC;

double* _A_tile;
double* _B_tile;
double* _C_tile;
double* _A;
double* _B;

  qmckl_tile_struct A_tile;
  qmckl_tile_struct B_tile;
  qmckl_tile_struct C_tile;

} qmckl_context_struct;

//typedef qmckl_context_struct* qmckl_context_struct_p;

void init_dims_avx512();
void init_dims_avx2();

int dgemm_main_tiled_avx2_NN(qmckl_context context, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

qmckl_exit_code dgemm_main_packed_avx2(qmckl_context context, double alpha, 
			  double *A, double *B, double beta, double *C);

#endif
