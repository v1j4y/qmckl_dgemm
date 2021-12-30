//#include "kernel.h"
#include "kernel_avx2_16regs.h"
#include "kernel_avx2_8regs.h"
#include "kernel_sse2_8regs.h"
//#include "kernel_avx2_32regs.h"

static double *_A_tile = NULL;
static double *_B_tile = NULL;
static double *_A = NULL; //[MC*KC] __attribute__ ((aligned(64)));
static double *_B = NULL; //[NC*KC] __attribute__ ((aligned(64)));

#ifdef DEFINE_QMCKL_MNK
int64_t qmckl_M;
int64_t qmckl_N;
int64_t qmckl_K;
int64_t MC;
int64_t NC;
int64_t KC;
#else
extern int64_t qmckl_M;
extern int64_t qmckl_N;
extern int64_t qmckl_K;
extern int64_t MC;
extern int64_t NC;
extern int64_t KC;
#endif

void init_dims_avx512();
void init_dims_avx2();
void init_dims_avx2_input(int64_t DIM_M, int64_t DIM_N, int64_t DIM_K);

int dgemm_main_tiled(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int dgemm_main_tiled_avx2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int dgemm_main_tiled_avx2_8regs(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);


int dgemm_main_tiled_sse2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);

int tile_matrix(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile);

int tile_matrix_general(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile);

int dgemm_naive(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC);
