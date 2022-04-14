#ifndef __QMCKL_DGEMM_PRIVATE_H
#define __QMCKL_DGEMM_PRIVATE_H
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "kernel_avx2_16regs.h"

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
typedef int64_t qmckl_packed_matrix;
typedef int32_t qmckl_exit_code;

typedef struct qmckl_packed_struct{
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
} qmckl_packed_struct;

typedef struct qmckl_context_struct{

} qmckl_context_struct;

#endif
