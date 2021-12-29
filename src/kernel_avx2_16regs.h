#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void dgemm_macro_kernel_avx2_16regs(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B);
//void dgemm_macro_kernel_avx2_12x4_16regs(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B);
