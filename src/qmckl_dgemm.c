#include "utils.h"
#include "qmckl_dgemm.h"
#include "qmckl_dgemm_private.h"

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

qmckl_context qmckl_tile_matrix_create() {

  qmckl_tile_struct* const tile_mat = (qmckl_tile_struct* const) malloc (sizeof(qmckl_tile_struct));

  if (tile_mat == NULL) {
    return QMCKL_NULL_CONTEXT;
  }

  /* Set all pointers and values to NULL */
  {
    memset(tile_mat, 0, sizeof(qmckl_tile_struct));
  }

  return (qmckl_tile_matrix) tile_mat;
}

qmckl_exit_code qmckl_init_pack(qmckl_context context, qmckl_tile_matrix tile_matrix, unsigned char mType, int64_t M8, int64_t N8, int64_t K8) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  if(mType == 'A' || mType == 'a'){
    // Initialize Tile data for A
    ctx->A_tile.Nt = K8;
    ctx->A_tile.NCt = K8;
    if(ctx->A_tile.Nt > 1152) ctx->A_tile.NCt = ctx->A_tile.Nt/2;
    if((M8 % MR2) != 0){

      ctx->A_tile.Mt = (int64_t)((M8/MR2)+1)*MR2;
      ctx->A_tile.MCt = ctx->A_tile.Mt;
      if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
      //if(ctx->A_tile.Mt < 512) ctx->A_tile.MCt = ctx->A_tile.Mt;
      //else ctx->A_tile.MCt = MR2*64;
    }
    else{
      ctx->A_tile.Mt = (int64_t)M8;
      ctx->A_tile.MCt = ctx->A_tile.Mt;
      if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
    }
  }
  else if(mType == 'B' || mType == 'b'){
    ctx->B_tile.Mt = K8;
    ctx->B_tile.MCt = K8;
    if(ctx->B_tile.Mt > 1152) ctx->B_tile.MCt = ctx->B_tile.Mt/2;
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

qmckl_exit_code qmckl_pack_matrix(qmckl_context context, unsigned char mType, int64_t M8, int64_t N8, double *Ain, int64_t LDA) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  if(mType == 'A' || mType == 'a'){
    int64_t mb = ctx->A_tile.Mt / ctx->A_tile.MCt;
    int64_t kb = ctx->A_tile.Nt / ctx->A_tile.NCt;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCKC = ctx->A_tile.MCt*ctx->A_tile.NCt;
    int64_t MCmax, NCmax;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;
    int64_t Min = M8;
    //printf("in A\n");
    //for(i=0;i<M8;++i){
    //  for(j=0;j<N8;++j){
    //	printf(" %3.2f ", Ain[i*N8 + j]);
    //  }
    //  printf("\n");
    //}
    //printf("M8=%ld N8=%ld LDA=%ld\n",M8,N8,LDA);
    //printf("Mt=%ld Nt=%ld\n",ctx->A_tile.Mt,ctx->A_tile.Nt);
    //printf("MC=%ld KC=%ld\n",ctx->A_tile.MCt,ctx->A_tile.NCt);

    // Initialize buffers
    if( ctx->_A == NULL) {
      ctx->_A  = (double *)aligned_alloc(64, ctx->A_tile.MCt*ctx->A_tile.NCt * sizeof(double));
      for(i=0;i<ctx->A_tile.MCt*ctx->A_tile.NCt;++i){
	ctx->_A[i]=0.0;
      }
    }
  
    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    if( ctx->A_tile.data == NULL) ctx->A_tile.data = (double *)aligned_alloc(64, ctx->A_tile.Mt*ctx->A_tile.Nt*2 * sizeof(double));
    for(k=0;k<kb;++k) {
      int64_t kc = ctx->A_tile.NCt;

      idxk = k * kc;

      for(j=0;j<mb;++j) {
	if ( (j+1)*ctx->A_tile.MCt > Min){
	  MCmax = Min - (j+0)*ctx->A_tile.MCt;
	}
	else{
	  MCmax = ctx->A_tile.MCt;
	}
	packA_general(context, kc, MCmax, &Ain[idxk + j*ctx->A_tile.MCt*LDA], LDA, 1, ctx->_A);

	// Write to tiled matrix to A
	//total=0;
	for(ii=0;ii<MCKC;++ii) {
	  ctx->A_tile.data[i_tile_a * (MCKC) + ii] = ctx->_A[ii];
	  //total = total ^ _A[ii];
	}
	//ctx->A_tile.data[i_tile_a * (MCKC) + 0] = total;
	i_tile_a += 1;
      }
    }
  }
  else if(mType == 'B' || mType == 'b'){
    int64_t nb = ctx->B_tile.Nt / ctx->B_tile.NCt;
    int64_t kb = ctx->B_tile.Mt / ctx->B_tile.MCt;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t NCKC = ctx->B_tile.NCt*ctx->B_tile.MCt;
    int64_t MCmax, NCmax;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;
    int64_t Nin = N8;

    // Initialize buffers
    if( ctx->_B == NULL) {
      ctx->_B  = (double *)aligned_alloc(64, ctx->B_tile.NCt*ctx->B_tile.MCt * sizeof(double));
      for(i=0;i<ctx->B_tile.NCt*ctx->B_tile.MCt;++i){
	ctx->_B[i]=0.0;
      }
    }

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    if( ctx->B_tile.data == NULL) ctx->B_tile.data = (double *)aligned_alloc(64, ctx->B_tile.Mt*ctx->B_tile.Nt*2 * sizeof(double));

    for(i=0;i<nb;++i) {
      if ( (i+1)*ctx->B_tile.NCt > Nin){
        NCmax = Nin - (i+0)*ctx->B_tile.NCt;
      }
      else{
        NCmax = ctx->B_tile.NCt;
      }
      for(k=0;k<kb;++k) {
	int64_t kc = ctx->B_tile.MCt;
	packB_general(context, kc, NCmax, &Ain[k*ctx->B_tile.MCt*LDA + i*ctx->B_tile.NCt], LDA, 1, ctx->_B);

	// Write to tiled matrix to B
	for(ii=0;ii<NCKC;++ii) {
	  ctx->B_tile.data[i_tile_b * (NCKC) + ii] = ctx->_B[ii];
	}
	i_tile_b += 1;
      }
    }
  }
  else if(mType == 'C' || mType == 'c'){

    int i,j,k,ii;
    //printf("in C\n");
    //for(i=0;i<M8;++i){
    //  for(j=0;j<N8;++j){
    //	printf(" %3.2f ", Ain[i*N8 + j]);
    //  }
    //  printf("\n");
    //}
    //printf("Mt=%ld Nt=%ld\n",ctx->C_tile.Mt,ctx->C_tile.Nt);

    // Initialize C_tile
    if( ctx->C_tile.data == NULL) ctx->C_tile.data = (double *)aligned_alloc(64, ctx->C_tile.Mt*ctx->C_tile.Nt   * sizeof(double));

    // Initialize C_tile
    for(i=0;i<ctx->C_tile.Mt*ctx->C_tile.Nt;++i){
      ctx->C_tile.data[i] = 0.0;
    }
  }
  else{
    printf("Wrong mType in qmckl_pack_matrix. mType=%c\n",mType);
  }
  
  return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_dgemm_tiled_avx2_nn(qmckl_context context, double *A, int64_t incRowA,
                                                double *B, int64_t incRowB,
                                                double *C, int64_t incRowC) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int64_t mb = ctx->A_tile.Mt / ctx->A_tile.MCt;
  int64_t nb = ctx->B_tile.Nt / ctx->B_tile.NCt;
  int64_t kb = ctx->A_tile.Nt / ctx->A_tile.NCt;
  int64_t idxi = 0;
  int64_t idxk = 0;
  int64_t nmbnb = 0;
  int64_t nmbnb_prev = 0;
  int64_t mc = ctx->A_tile.MCt;
  int64_t nc = ctx->B_tile.NCt;
  int64_t kc = ctx->A_tile.NCt;
  int64_t MCNC = mc*nc;
  size_t szeA = mc*kc*sizeof(double);
  size_t szeB = nc*kc*sizeof(double);
  int i,j,k,imb;
  int MCKC = mc*kc;
  int NCKC = nc*kc;

  int64_t i_tile_b, i_tile_a;
  double *A_tile_p __attribute__ ((aligned(64)));
  double *B_tile_p __attribute__ ((aligned(64)));
  double *C_tile_p __attribute__ ((aligned(64)));

  // Initialize indices
  i_tile_b = 0;
  i_tile_a = 0;

  B_tile_p = ctx->B_tile.data;
  A_tile_p = ctx->A_tile.data;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = ctx->A_tile.data;
        //B_tile_p = _B_tile + i*kb*(NCKC);
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {

            idxi = i * nc * incRowC;
            idxk = k * kc;

            //B_tile_p = _B_tile + i_tile_b * (NCKC);
            C_tile_p = ctx->C_tile.data + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_16regs(mc, kc, nc, C_tile_p, incRowC, 1, A_tile_p, B_tile_p);
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


qmckl_exit_code qmckl_unpack_matrix(qmckl_context context, double *B, int64_t M, int64_t N) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  int64_t mc = ctx->C_tile.MCt;
  int64_t nc = ctx->C_tile.NCt;
  int64_t mb = M / mc;
  int64_t nb = N / nc;
  int64_t mp = mc / MR;
  int64_t np = nc / NR;
  int i,j;
  for(i=0;i<N;++i) {
    int64_t lnc = ( i / nc );
    int64_t lnr = ( i - lnc * nc ) / NR;
    int64_t l   = ( ( i - lnc * nc ) - ( lnr * NR ) );
    for(j=0;j<M;++j) {
      int64_t kmc = ( j / mc );
      int64_t kmr = ( j - kmc * mc ) / MR;
      int64_t k   = ( ( j - kmc * mc ) - ( kmr * MR ) );
      B[i + j*N] = ctx->C_tile.data[(mc*nc)*(lnc*mb) + (mc*nc)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k];
      }
  }
  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_tile_matrix_destroy(qmckl_tile_matrix tile_matrix){

  qmckl_tile_struct* const ctx = (qmckl_tile_struct* const) tile_matrix;

  // Free data
  if( ctx->data != NULL){
    free(ctx->data);
    ctx->data = NULL;
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

  // Free tiles
  if( ctx->A_tile.data != NULL){
    free(ctx->A_tile.data);
    ctx->A_tile.data = NULL;
  }
  if( ctx->B_tile.data != NULL){
    free(ctx->B_tile.data);
    ctx->B_tile.data = NULL;
  }
  if( ctx->C_tile.data != NULL){
    free(ctx->C_tile.data);
    ctx->C_tile.data = NULL;
  }

  return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_dgemm_tiled_NN(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin,
				     double *A, int64_t incRowA,
				     double *B, int64_t incRowB,
				     double *C, int64_t incRowC) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_tile_matrix const tile_matrix;

  // Init memory
  qmckl_init_pack(context, tile_matrix, 'A', Min, Nin, Kin);
  qmckl_init_pack(context, tile_matrix, 'B', Min, Nin, Kin);
  qmckl_init_pack(context, tile_matrix, 'C', Min, Nin, Kin);

  // Tile A and B
  qmckl_pack_matrix(context, 'A', Min, Kin, A, incRowA);
  qmckl_pack_matrix(context, 'B', Kin, Nin, B, incRowB);
  qmckl_pack_matrix(context, 'C', Min, Nin, C, incRowB);



  // Call DGEMM kernel
  qmckl_dgemm_tiled_avx2_nn(context, A, incRowA,
             B, incRowB,
             ctx->C_tile.data, incRowC);

  // Unpacking
  qmckl_unpack_matrix(context, C, Min, Nin);

  //// Free memory
  //qmckl_context_destroy(context);

  return QMCKL_SUCCESS;
}
