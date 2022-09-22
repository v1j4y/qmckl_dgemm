#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void dgemm_macro_kernel_sse2_8regs(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B);
