#include "utils.h"
#include "qmckl_dgemm.h"
#include "qmckl_dgemm_private.h"

// Global context
qmckl_context_struct ctxt = {.qmckl_M=0, .qmckl_N=0, .qmckl_K=0, .MC=0, .NC=0, .KC=0, 
                       ._A_tile=NULL, ._B_tile=NULL, ._C_tile=NULL,
                       ._A=NULL, ._B=NULL};
//qmckl_context_struct_p ctxtp = &ctxt;

qmckl_context qmckl_context_create() {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) malloc (sizeof(qmckl_context_struct));

  if (ctx == NULL) {
    return QMCKL_NULL_CONTEXT;
  }

  /* Set all pointers and values to NULL */
  {
    memset(ctx, 0, sizeof(qmckl_context_struct));
  }

  return (qmckl_context) ctx;
}

qmckl_exit_code qmckl_init_tile(qmckl_context context, unsigned char mType, int64_t M8, int64_t N8, int64_t K8) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  if(mType == 'A' || mType == 'a'){
    // Initialize Tile data for A
    ctx->A_tile.Nt = K8;
    if((M8 % MR2) != 0){

      ctx->A_tile.Mt = (int64_t)((M8/MR2)+1)*MR2;
      ctx->A_tile.MCt = ctx->A_tile.Mt;
      if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
    }
    else{
      ctx->A_tile.Mt = (int64_t)M8;
      ctx->A_tile.MCt = ctx->A_tile.Mt;
      if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
    }
  }
  else if(mType == 'B' || mType == 'b'){
    ctx->B_tile.Mt = K8;
    if((N8 % NR2) != 0){
      ctx->B_tile.Nt = (int64_t)((N8/NR2)+1)*NR2;
      ctx->B_tile.NCt = ctx->B_tile.Nt;
      if(ctx->B_tile.Nt > 1152) ctx->B_tile.NCt = ctx->B_tile.Nt/2;
    }
    else{
      ctx->B_tile.Nt = (int64_t)N8;
      ctx->B_tile.NCt = ctx->B_tile.Nt;
      if(ctx->B_tile.Nt > 1152) ctx->B_tile.NCt = ctx->B_tile.Nt/2;
    }
  }
  else if(mType == 'C' || mType == 'c'){
    if((M8 % MR2) != 0){

      ctx->C_tile.Mt = (int64_t)((M8/MR2)+1)*MR2;
      ctx->C_tile.MCt = ctx->C_tile.Mt;
      if(ctx->C_tile.Mt > 1152) ctx->C_tile.MCt = ctx->C_tile.Mt/2;
    }
    else{
      ctx->C_tile.Mt = (int64_t)M8;
      ctx->C_tile.MCt = ctx->C_tile.Mt;
      if(ctx->C_tile.Mt > 1152) ctx->C_tile.MCt = ctx->C_tile.Mt/2;
    }
    if((N8 % NR2) != 0){
      ctx->C_tile.Nt = (int64_t)((N8/NR2)+1)*NR2;
      ctx->C_tile.NCt = ctx->C_tile.Nt;
      if(ctx->C_tile.Nt > 1152) ctx->C_tile.NCt = ctx->C_tile.Nt/2;
    }
    else{
      ctx->C_tile.Nt = (int64_t)N8;
      ctx->C_tile.NCt = ctx->C_tile.Nt;
      if(ctx->C_tile.Nt > 1152) ctx->C_tile.NCt = ctx->C_tile.Nt/2;
    }
  }
  else{
    printf("Wrong mType in qmckl_init_tile. mType=%c\n",mType);
  }

  return QMCKL_SUCCESS;
}

qmckl_exit_code init_dims_avx2_input(qmckl_context context, int64_t DIM_M, int64_t DIM_N, int64_t DIM_K) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
    //qmckl_M = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_N = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_K = (int64_t *)malloc(1 * sizeof(int64_t));
    //
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int MR2NR2 = MR2*NR2;

  ctx->KC = DIM_K;

  if((DIM_M % MR2) != 0){

    ctx->qmckl_M = (int64_t)((DIM_M/MR2)+1)*MR2;
    ctx->MC = ctx->qmckl_M;
    if(ctx->qmckl_M > 1152) ctx->MC = ctx->qmckl_M/2;
  }
  else{
    ctx->qmckl_M = (int64_t)DIM_M;
    ctx->MC = ctx->qmckl_M;
    if(ctx->qmckl_M > 1152) ctx->MC = ctx->qmckl_M/2;
  }

  if((DIM_K % ctx->KC) != 0){
    ctx->qmckl_K = (int64_t)((DIM_K/ctx->KC)+1)*ctx->KC;
  }
  else{
    ctx->qmckl_K = (int64_t)DIM_K;
  }

  if((DIM_N % NR2) != 0){
    ctx->qmckl_N = (int64_t)((DIM_N/NR2)+1)*NR2;
    ctx->NC = ctx->qmckl_N;
    if(ctx->qmckl_N > 1152) ctx->NC = ctx->qmckl_N/2;
  }
  else{
    ctx->qmckl_N = (int64_t)DIM_N;
    ctx->NC = ctx->qmckl_N;
    if(ctx->qmckl_N > 1152) ctx->NC = ctx->qmckl_N/2;
  }
  printf("(AVX2) M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)ctx->qmckl_M,(long)ctx->qmckl_K,(long)ctx->qmckl_N,(long)ctx->MC,(long)ctx->KC,(long)ctx->NC);
  return QMCKL_SUCCESS;
}


qmckl_exit_code dgemm_main_tiled_avx2(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int64_t mb = Min / ctx->MC;
  int64_t nb = Nin / ctx->NC;
  int64_t kb = Kin / ctx->KC;
  int64_t idxi = 0;
  int64_t idxk = 0;
  int64_t nmbnb = 0;
  int64_t nmbnb_prev = 0;
  int64_t MCNC = ctx->MC*ctx->NC;
  size_t szeA = ctx->MC*ctx->KC*sizeof(double);
  size_t szeB = ctx->NC*ctx->KC*sizeof(double);
  int i,j,k,imb;
  int MCKC = ctx->MC*ctx->KC;
  int NCKC = ctx->NC*ctx->KC;

  int64_t i_tile_b, i_tile_a;
  double *A_tile_p __attribute__ ((aligned(64)));
  double *B_tile_p __attribute__ ((aligned(64)));
  double *C_tile_p __attribute__ ((aligned(64)));

  // Initialize indices
  i_tile_b = 0;
  i_tile_a = 0;

  B_tile_p = ctx->_B_tile;
  A_tile_p = ctx->_A_tile;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = ctx->_A_tile;
        //B_tile_p = _B_tile + i*kb*(NCKC);
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = ctx->KC;

            idxi = i * ctx->NC * incRowC;
            idxk = k * ctx->KC * incColA;

            //B_tile_p = _B_tile + i_tile_b * (NCKC);
            C_tile_p = ctx->_C_tile + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_16regs(ctx->MC, ctx->KC, ctx->NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
                A_tile_p += (MCKC);
                C_tile_p +=  MCNC;
                //nmbnb = nmbnb + 1;
                i_tile_a += 1;
            }
            //if(k < (kb-1)) nmbnb = nmbnb_prev;
            B_tile_p += (NCKC);
            i_tile_b += 1;
        }
    }
//}

    return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_unpack(qmckl_context context, double *B, int64_t M, int64_t N) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int64_t mb = M / ctx->MC;
  int64_t nb = N / ctx->NC;
  int64_t mp = ctx->MC / MR;
  int64_t np = ctx->NC / NR;
  int i,j;
  for(i=0;i<N;++i) {
      for(j=0;j<M;++j) {
        int64_t kmc = ( j / ctx->MC );
        int64_t lnc = ( i / ctx->NC );
        int64_t kmr = ( j - kmc * ctx->MC ) / MR;
        int64_t lnr = ( i - lnc * ctx->NC ) / NR;
        int64_t k   = ( ( j - kmc * ctx->MC ) - ( kmr * MR ) );
        int64_t l   = ( ( i - lnc * ctx->NC ) - ( lnr * NR ) );
        B[j + i*M] = ctx->_C_tile[(ctx->MC*ctx->NC)*(lnc*mb) + (ctx->MC*ctx->NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k];
      }
  }
  return QMCKL_SUCCESS;
}

qmckl_exit_code tile_matrix_general(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int64_t mb = ctx->qmckl_M / ctx->MC;
  int64_t nb = ctx->qmckl_N / ctx->NC;
  int64_t kb = ctx->qmckl_K / ctx->KC;
  int64_t idxi = 0;
  int64_t idxk = 0;
  int64_t nmbnb = 0;
  int64_t nmbnb_prev = 0;
  int64_t MCNC = ctx->MC*ctx->NC;
  int64_t NCKC = ctx->NC*ctx->KC;
  int64_t MCKC = ctx->MC*ctx->KC;
  int64_t MCmax, NCmax;
  int64_t i_tile_b, i_tile_a;
  int i,j,k,ii;

//atic double _A[MC*KC] __attribute__ ((aligned(64)));
//atic double _B[NC*KC] __attribute__ ((aligned(64)));
  if( ctx->_A == NULL) ctx->_A  = (double *)aligned_alloc(64, ctx->MC*ctx->KC * sizeof(double));
  if( ctx->_B == NULL) ctx->_B  = (double *)aligned_alloc(64, ctx->NC*ctx->KC * sizeof(double));
  if( ctx->_A_tile == NULL) ctx->_A_tile = (double *)aligned_alloc(64, ctx->qmckl_M*ctx->qmckl_K*2 * sizeof(double));
  if( ctx->_B_tile == NULL) ctx->_B_tile = (double *)aligned_alloc(64, ctx->qmckl_N*ctx->qmckl_K*2 * sizeof(double));
  if( ctx->_B_tile == NULL) ctx->_B_tile = (double *)aligned_alloc(64, ctx->qmckl_N*ctx->qmckl_K*2 * sizeof(double));
  if( ctx->_C_tile == NULL) ctx->_C_tile = (double *)aligned_alloc(64, ctx->qmckl_M*ctx->qmckl_N   * sizeof(double));

  // Initialize indices
  i_tile_b = 0;
  i_tile_a = 0;

  for(i=0;i<nb;++i) {
      if ( (i+1)*ctx->NC > Nin){
        NCmax = Nin - (i+0)*ctx->NC;
      }
      else{
        NCmax = ctx->NC;
      }
      for(k=0;k<kb;++k) {
          int64_t kc = ctx->KC;
          packB_general(context, kc, NCmax, &B[k*ctx->KC*incRowB + i*ctx->NC*incColB], incRowB, incColB, ctx->_B);

          // Write to tiled matrix to B
          for(ii=0;ii<NCKC;++ii) {
            ctx->_B_tile[i_tile_b * (NCKC) + ii] = ctx->_B[ii];
          }
          i_tile_b += 1;
      }
  }

  for(k=0;k<kb;++k) {
      int64_t kc = ctx->KC;

      idxk = k * ctx->KC * incColA;

      for(j=0;j<mb;++j) {
          if ( (j+1)*ctx->MC > Min){
            MCmax = Min - (j+0)*ctx->MC;
          }
          else{
            MCmax = ctx->MC;
          }
          packA_general(context, kc, MCmax, &A[idxk + j*ctx->MC*incRowA], incRowA, incColA, ctx->_A);

          // Write to tiled matrix to A
          for(ii=0;ii<MCKC;++ii) {
            ctx->_A_tile[i_tile_a * (MCKC) + ii] = ctx->_A[ii];
          }
          i_tile_a += 1;
      }
  }

  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_context_destroy(qmckl_context context){

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  if( ctx->_A != NULL){
    free(ctx->_A);
    ctx->_A = NULL;
  }
  if( ctx->_B != NULL){
    free(ctx->_B);
    ctx->_B = NULL;
  }
  if( ctx->_A_tile != NULL){
    free(ctx->_A_tile);
    ctx->_A_tile = NULL;
  }
  if( ctx->_B_tile != NULL){
    free(ctx->_B_tile);
    ctx->_B_tile = NULL;
  }
  if( ctx->_B_tile != NULL){
    free(ctx->_B_tile);
    ctx->_B_tile = NULL;
  }
  if( ctx->_C_tile != NULL){
    free(ctx->_C_tile);
    ctx->_C_tile = NULL;
  }
  return QMCKL_SUCCESS;
}


qmckl_exit_code dgemm_main_tiled(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  // Init memory
  init_dims_avx2_input(context, Min, Nin, Kin);

  // Tile A and B
  tile_matrix_general(context, Min, Nin, Kin, A, incRowA, incColA,
             B, incRowB, incColB,
             C, incRowC, incColC, ctx->_A_tile, ctx->_B_tile);


  // Call DGEMM kernel
  dgemm_main_tiled_avx2(context, ctx->qmckl_M, ctx->qmckl_N, ctx->qmckl_K, A, incRowA, incColA,
             B, incRowB, incColB,
             ctx->_C_tile, incRowC, incColC);

  // Unpacking
  qmckl_unpack(context, C, Min, Nin);

  // Free memory
  qmckl_context_destroy(context);

  return QMCKL_SUCCESS;
}
