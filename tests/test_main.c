#include <stdio.h>

#include "mkl.h"

#include "utils.h"
#include "qmckl_dgemm.h"

//int main(int argc, char *argv[]) {
int main() {

  double *A;
  double *B;
  double *C;
  double *CUnpack;
  double *ABlas;
  double *BBlas;
  double *DBlas;
  int64_t DIM_M, DIM_N, DIM_K;
  int64_t M, N, K;
  int64_t MBlas, NBlas, KBlas;
  int64_t incColA = 1;
  int64_t incColB = 1;
  int64_t incColC = 1;
  int64_t iterM, iterN, iterK;
  int64_t rep = 0;
  
  int64_t DIM_M_MAX =MR2*2;
  int64_t DIM_N_MAX =NR2*2;
  int64_t DIM_K_MAX =10;

  for(iterM = 1;iterM <= DIM_M_MAX; ++iterM){
    DIM_M = iterM;
    for(iterN = 1;iterN <= DIM_N_MAX; ++iterN){
      DIM_N = iterN;
      for(iterK = 1;iterK <= DIM_K_MAX; ++iterK){
	DIM_K = iterK;
    
  
  qmckl_context context = qmckl_context_create();
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  //init_dims_avx2_input(context, DIM_M, DIM_N, DIM_K);
  qmckl_init_pack(context, 'A', DIM_M, DIM_N, DIM_K);
  qmckl_init_pack(context, 'B', DIM_M, DIM_N, DIM_K);
  qmckl_init_pack(context, 'C', DIM_M, DIM_N, DIM_K);
  
  //M = qmckl_M;
  //N = qmckl_N;
  //K = qmckl_K;
  M = DIM_M;
  N = DIM_N;
  K = DIM_K;
  MBlas = M;
  NBlas = N;
  KBlas = K;
  printf("\n----------------------------------\n");
  printf("(%d, %d, %d) | M=%ld K=%ld N=%ld | ",iterM, iterN, iterK, (long)M,(long)K,(long)N);
  
  int64_t incRowA = K;
  int64_t incRowB = N;
  int64_t incRowC = ctx->C_tile.Nt;
  
  A = (double *)malloc( M * K * sizeof(double));
  B = (double *)malloc( K * N * sizeof(double));
  C = (double *)aligned_alloc( 64, ctx->C_tile.Mt * ctx->C_tile.Nt * sizeof(double));
  CUnpack = (double *)aligned_alloc( 64, ctx->C_tile.Mt * ctx->C_tile.Nt * sizeof(double));
  
  ABlas = (double *)malloc( MBlas * KBlas * sizeof(double));
  BBlas = (double *)malloc( KBlas * NBlas * sizeof(double));
  DBlas = (double *)malloc( MBlas * NBlas * sizeof(double));
  
  fill_matrix_random(A, M,K);
  fill_matrix_random(B, K, N);
  fill_matrix_zeros  (C, M*N);
  
  copy_matrix(ABlas, A, MBlas,KBlas);
  copy_matrix(BBlas, B, KBlas,NBlas);
  fill_matrix_zeros  (DBlas, MBlas*NBlas);
  
  int i,j=rep;
  
  qmckl_pack_matrix(context, 'A', M, K, A, incRowA);
  qmckl_pack_matrix(context, 'B', K, N, B, incRowB);
  qmckl_pack_matrix(context, 'C', M, N, C, incRowC);
  
  qmckl_dgemm_tiled_avx2_nn(context, A, incRowA,
			   B, incRowB,
			   C, incRowC);
  
  const int MB = MBlas;
  const int NB = NBlas;
  const int KB = KBlas;
  const double alpha=1.0;
  const double beta=0.0;
  double *ABlasp;
  double *BBlasp;
  
  // Get size of packed A
  size_t ABlasp_size = dgemm_pack_get_size("A",&MB,&NB,&KB);
  ABlasp = (double *)mkl_malloc(ABlasp_size,64);
  
  // Pack
  dgemm_pack("A","T",&MB,&NB,&KB,&alpha,ABlas,&KB,ABlasp);
  
  // Get size of packed B
  size_t BBlasp_size = dgemm_pack_get_size("B",&MB,&NB,&KB);
  BBlasp = (double *)mkl_malloc(BBlasp_size,64);
  
  // Pack
  dgemm_pack("B","T",&MB,&NB,&KB,&alpha,BBlas,&NB,BBlasp);
  
  dgemm_compute("P","P",&MB,&NB,&KB,ABlasp,&KB,BBlasp,&NB,&beta,DBlas,&MB);
  
  //print_matrix(DBlas, M, N);
  //printf("\n-------------diff-----------------\n");
  //print_diff_matrix_AT_B(C,D, M, N);
  //print_diff_matrix_ASer_BT(context, C,DBlas, M, N);
  qmckl_unpack_matrix(context, CUnpack, M, N);
  //print_diff_matrix(CUnpack,DBlas, M, N);
  qmckl_exit_code rc = get_diff_matrix_ABT(CUnpack,DBlas, M, N);
  if(rc == QMCKL_FAILURE){
    printf(" QMCKL_FAILURE !\n");
    return QMCKL_FAILURE;
  }
  else{
    printf(" QMCKL_SUCCESS !\n");
  }
  //print_matrix(CUnpack, M, N);
  printf("\n----------------------------------\n");
  
  mkl_free(ABlasp);
  mkl_free(BBlasp);
  qmckl_context_destroy(context);
  free(A);
  free(B);
  free(C);
  free(CUnpack);
  free(ABlas);
  free(BBlas);
  free(DBlas);
      }
    }
  }

  return QMCKL_SUCCESS;
}
