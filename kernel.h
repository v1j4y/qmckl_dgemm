#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


void dgemm_kernel(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC);
void dgemm_kernel_sse_asm(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC);
void dgemm_macro_kernel(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B);
