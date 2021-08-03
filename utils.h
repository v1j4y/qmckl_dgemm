#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if !defined(MAT_DIM)
#define MAT_DIM 1024
#endif

#if !defined(MC)
#define MC 512
#endif

#if !defined(NC)
#define NC 512
#endif

#if !defined(KC)
#define KC 512
#endif

#if !defined(MR)
#define MR 4
#endif

#if !defined(NR)
#define NR 4
#endif

static double _A[MC*KC] __attribute__ ((aligned(64)));
static double _B[NC*KC] __attribute__ ((aligned(64)));

unsigned long long rdtsc(void);
void fill_matrix_ones(double *A, int64_t dim);
void fill_matrix_zeros(double *A, int64_t dim);
void print_matrix(double *A, int64_t M, int64_t N);
void packA(int64_t kc, double *A, int64_t incRowA, int64_t incColA, double *buffer);
void packB(int64_t kc, double *B, int64_t incRowB, int64_t incColB, double *buffer);
