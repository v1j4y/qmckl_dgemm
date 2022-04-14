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

qmckl_context qmckl_packed_matrix_create() {

  qmckl_packed_struct* const packed_mat = (qmckl_packed_struct* const) malloc (sizeof(qmckl_packed_struct));

  if (packed_mat == NULL) {
    return QMCKL_NULL_CONTEXT;
  }

  /* Set all pointers and values to NULL */
  {
    memset(packed_mat, 0, sizeof(qmckl_packed_struct));
  }

  return (qmckl_packed_matrix) packed_mat;
}

qmckl_exit_code qmckl_init_pack(qmckl_context context, qmckl_packed_matrix packed_matrix, unsigned char mType, int64_t M8, int64_t N8, int64_t K8) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  qmckl_packed_struct* const pmat = (qmckl_packed_struct* const) packed_matrix;

  if(mType == 'A' || mType == 'a'){
    // Initialize Tile data for A
    //ctx->A_tile.Nt = K8;
    //ctx->A_tile.NCt = K8;
    pmat->Nt = K8;
    pmat->NCt = K8;
    //if(ctx->A_tile.Nt > 1152) ctx->A_tile.NCt = ctx->A_tile.Nt/2;
    if(pmat->Nt > 1152) pmat->NCt = pmat->Nt/2;
    if((M8 % MR2) != 0){

      //ctx->A_tile.Mt = (int64_t)((M8/MR2)+1)*MR2;
      //ctx->A_tile.MCt = ctx->A_tile.Mt;
      //if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
      //if(ctx->A_tile.Mt < 512) ctx->A_tile.MCt = ctx->A_tile.Mt;
      //else ctx->A_tile.MCt = MR2*64;
      pmat->Mt = (int64_t)((M8/MR2)+1)*MR2;
      pmat->MCt = pmat->Mt;
      if(pmat->Mt > 1152) pmat->MCt = pmat->Mt/2;
    }
    else{
      //ctx->A_tile.Mt = (int64_t)M8;
      //ctx->A_tile.MCt = ctx->A_tile.Mt;
      //if(ctx->A_tile.Mt > 1152) ctx->A_tile.MCt = ctx->A_tile.Mt/2;
      pmat->Mt = (int64_t)M8;
      pmat->MCt = pmat->Mt;
      if(pmat->Mt > 1152) pmat->MCt = pmat->Mt/2;
    }
  }
  else if(mType == 'B' || mType == 'b'){
    //ctx->B_tile.Mt = K8;
    //ctx->B_tile.MCt = K8;
    //if(ctx->B_tile.Mt > 1152) ctx->B_tile.MCt = ctx->B_tile.Mt/2;
    //if((N8 % NR2) != 0){
    //  ctx->B_tile.Nt = (int64_t)((N8/NR2)+1)*NR2;
    //  ctx->B_tile.NCt = ctx->B_tile.Nt;
    //  if(ctx->B_tile.Nt > 1152) ctx->B_tile.NCt = ctx->B_tile.Nt/2;
    //}
    //else{
    //  ctx->B_tile.Nt = (int64_t)N8;
    //  ctx->B_tile.NCt = ctx->B_tile.Nt;
    //  if(ctx->B_tile.Nt > 1152) ctx->B_tile.NCt = ctx->B_tile.Nt/2;
    //}

    // Tile
    pmat->Mt = K8;
    pmat->MCt = K8;
    if(pmat->Mt > 1152) pmat->MCt = pmat->Mt/2;
    if((N8 % NR2) != 0){
      pmat->Nt = (int64_t)((N8/NR2)+1)*NR2;
      pmat->NCt = pmat->Nt;
      if(pmat->Nt > 1152) pmat->NCt = pmat->Nt/2;
    }
    else{
      pmat->Nt = (int64_t)N8;
      pmat->NCt = pmat->Nt;
      if(pmat->Nt > 1152) pmat->NCt = pmat->Nt/2;
    }
  }
  else if(mType == 'C' || mType == 'c'){
    //if((M8 % MR2) != 0){

    //  ctx->C_tile.Mt = (int64_t)((M8/MR2)+1)*MR2;
    //  ctx->C_tile.MCt = ctx->C_tile.Mt;
    //  if(ctx->C_tile.Mt > 1152) ctx->C_tile.MCt = ctx->C_tile.Mt/2;
    //}
    //else{
    //  ctx->C_tile.Mt = (int64_t)M8;
    //  ctx->C_tile.MCt = ctx->C_tile.Mt;
    //  if(ctx->C_tile.Mt > 1152) ctx->C_tile.MCt = ctx->C_tile.Mt/2;
    //}
    //if((N8 % NR2) != 0){
    //  ctx->C_tile.Nt = (int64_t)((N8/NR2)+1)*NR2;
    //  ctx->C_tile.NCt = ctx->C_tile.Nt;
    //  if(ctx->C_tile.Nt > 1152) ctx->C_tile.NCt = ctx->C_tile.Nt/2;
    //}
    //else{
    //  ctx->C_tile.Nt = (int64_t)N8;
    //  ctx->C_tile.NCt = ctx->C_tile.Nt;
    //  if(ctx->C_tile.Nt > 1152) ctx->C_tile.NCt = ctx->C_tile.Nt/2;
    //}

    // Tile
    if((M8 % MR2) != 0){

      pmat->Mt = (int64_t)((M8/MR2)+1)*MR2;
      pmat->MCt = pmat->Mt;
      if(pmat->Mt > 1152) pmat->MCt = pmat->Mt/2;
    }
    else{
      pmat->Mt = (int64_t)M8;
      pmat->MCt = pmat->Mt;
      if(pmat->Mt > 1152) pmat->MCt = pmat->Mt/2;
    }
    if((N8 % NR2) != 0){
      pmat->Nt = (int64_t)((N8/NR2)+1)*NR2;
      pmat->NCt = pmat->Nt;
      if(pmat->Nt > 1152) pmat->NCt = pmat->Nt/2;
    }
    else{
      pmat->Nt = (int64_t)N8;
      pmat->NCt = pmat->Nt;
      if(pmat->Nt > 1152) pmat->NCt = pmat->Nt/2;
    }
  }
  else{
    printf("Wrong mType in qmckl_init_tile. mType=%c\n",mType);
  }

  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_pack_matrix(qmckl_context context, qmckl_packed_matrix packed_matrix, unsigned char mType, int64_t M8, int64_t N8, double *Ain, int64_t LDA) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_packed_struct* const pmat = (qmckl_packed_struct* const) packed_matrix;

  if(mType == 'A' || mType == 'a'){
    int64_t mb = pmat->Mt / pmat->MCt;
    int64_t kb = pmat->Nt / pmat->NCt;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCKC = pmat->MCt*pmat->NCt;
    int64_t MCmax, NCmax;
    int64_t i_packed_b, i_packed_a;
    int i,j,k,ii;
    int64_t Min = M8;
    double *_A=NULL;

    // Initialize buffers
    if( _A == NULL) {
      _A  = (double *)aligned_alloc(64, pmat->MCt*pmat->NCt * sizeof(double));
      for(i=0;i<pmat->MCt*pmat->NCt;++i){
	_A[i]=0.0;
      }
    }
  
    // Tile
    // Initialize indices
    i_packed_b = 0;
    i_packed_a = 0;

    if( pmat->data == NULL) pmat->data = (double *)aligned_alloc(64, pmat->Mt*pmat->Nt* sizeof(double));

    for(k=0;k<kb;++k) {
      int64_t kc = pmat->NCt;

      idxk = k * kc;

      for(j=0;j<mb;++j) {
	if ( (j+1)*pmat->MCt > Min){
	  MCmax = Min - (j+0)*pmat->MCt;
	}
	else{
	  MCmax = pmat->MCt;
	}
	packA_general(context, packed_matrix, kc, MCmax, &Ain[idxk + j*pmat->MCt*LDA], LDA, 1, _A);

	// Write to tiled matrix to A
	for(ii=0;ii<MCKC;++ii) {
	  pmat->data[i_packed_a * (MCKC) + ii] = _A[ii];
	}
	i_packed_a += 1;
      }
    }
    free(_A);
  }
  else if(mType == 'B' || mType == 'b'){
    int64_t nb = pmat->Nt / pmat->NCt;
    int64_t kb = pmat->Mt / pmat->MCt;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t NCKC = pmat->NCt*pmat->MCt;
    int64_t MCmax, NCmax;
    int64_t i_packed_b, i_packed_a;
    int i,j,k,ii;
    int64_t Nin = N8;
    double *_B=NULL;

    // Initialize buffers
    if( _B == NULL) {
      _B  = (double *)aligned_alloc(64, pmat->NCt*pmat->MCt * sizeof(double));
      for(i=0;i<pmat->NCt*pmat->MCt;++i){
	_B[i]=0.0;
      }
    }

    // Tile
    i_packed_b = 0;
    i_packed_a = 0;

    if( pmat->data == NULL) pmat->data = (double *)aligned_alloc(64, pmat->Mt*pmat->Nt * sizeof(double));

    for(i=0;i<nb;++i) {
      if ( (i+1)*pmat->NCt > Nin){
        NCmax = Nin - (i+0)*pmat->NCt;
      }
      else{
        NCmax = pmat->NCt;
      }
      for(k=0;k<kb;++k) {
	int64_t kc = pmat->MCt;
	packB_general(context, packed_matrix, kc, NCmax, &Ain[k*pmat->MCt*LDA + i*pmat->NCt], LDA, 1, _B);

	// Write to tiled matrix to B
	for(ii=0;ii<NCKC;++ii) {
	  pmat->data[i_packed_b * (NCKC) + ii] = _B[ii];
	}
	i_packed_b += 1;
      }
    }
    free(_B);
  }
  else if(mType == 'C' || mType == 'c'){

    int i,j,k,ii;

    // Tile
    if( pmat->data == NULL) pmat->data = (double *)aligned_alloc(64, pmat->Mt*pmat->Nt   * sizeof(double));

    // Initialize C_tile
    for(i=0;i<pmat->Mt*pmat->Nt;++i){
      pmat->data[i] = 0.0;
    }
  }
  else{
    printf("Wrong mType in qmckl_pack_matrix. mType=%c\n",mType);
  }
  
  return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_dgemm_tiled_avx2_nn(qmckl_context context, qmckl_packed_matrix packed_matrix_A, int64_t incRowA,
                                                qmckl_packed_matrix packed_matrix_B, int64_t incRowB,
                                                qmckl_packed_matrix packed_matrix_C, int64_t incRowC) {

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_packed_struct* const pmatA = (qmckl_packed_struct* const) packed_matrix_A;
  qmckl_packed_struct* const pmatB = (qmckl_packed_struct* const) packed_matrix_B;
  qmckl_packed_struct* const pmatC = (qmckl_packed_struct* const) packed_matrix_C;

  int64_t mb = pmatA->Mt / pmatA->MCt;
  int64_t nb = pmatB->Nt / pmatB->NCt;
  int64_t kb = pmatA->Nt / pmatA->NCt;
  int64_t idxi = 0;
  int64_t idxk = 0;
  int64_t nmbnb = 0;
  int64_t nmbnb_prev = 0;
  int64_t mc = pmatA->MCt;
  int64_t nc = pmatB->NCt;
  int64_t kc = pmatA->NCt;
  int64_t MCNC = mc*nc;
  size_t szeA = mc*kc*sizeof(double);
  size_t szeB = nc*kc*sizeof(double);
  int i,j,k,imb;
  int MCKC = mc*kc;
  int NCKC = nc*kc;

  int64_t i_packed_b, i_packed_a;
  double *A_packed_p __attribute__ ((aligned(64)));
  double *B_packed_p __attribute__ ((aligned(64)));
  double *C_packed_p __attribute__ ((aligned(64)));

  // Initialize indices
  i_packed_b = 0;
  i_packed_a = 0;

  B_packed_p = pmatB->data;
  A_packed_p = pmatA->data;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_packed_a = 0;
        A_packed_p = pmatA->data;
        //B_packed_p = _B_tile + i*kb*(NCKC);
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {

            idxi = i * nc * incRowC;
            idxk = k * kc;

            //B_packed_p = _B_tile + i_packed_b * (NCKC);
            C_packed_p = pmatC->data + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_16regs(mc, kc, nc, C_packed_p, incRowC, 1, A_packed_p, B_packed_p);
                A_packed_p += (MCKC);
                C_packed_p +=  MCNC;
                //nmbnb = nmbnb + 1;
                i_packed_a += 1;
            }
            //if(k < (kb-1)) nmbnb = nmbnb_prev;
            B_packed_p += (NCKC);
            i_packed_b += 1;
        }
    }
//}

    return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_unpack_matrix(qmckl_context context, qmckl_packed_matrix packed_matrix, double *B, int64_t M, int64_t N) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_packed_struct* const pmat = (qmckl_packed_struct* const) packed_matrix;
  int64_t mc = pmat->MCt;
  int64_t nc = pmat->NCt;
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
      B[i + j*N] = pmat->data[(mc*nc)*(lnc*mb) + (mc*nc)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k];
      }
  }
  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_packed_matrix_destroy(qmckl_packed_matrix packed_matrix){

  qmckl_packed_struct* const pmat = (qmckl_packed_struct* const) packed_matrix;

  // Free data
  if( pmat->data != NULL){
    free(pmat->data);
    pmat->data = NULL;
  }

  return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_context_destroy(qmckl_context context){

  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;

  // Free tiles
  //if( ctx->A_tile.data != NULL){
  //  free(ctx->A_tile.data);
  //  ctx->A_tile.data = NULL;
  //}
  //if( ctx->B_tile.data != NULL){
  //  free(ctx->B_tile.data);
  //  ctx->B_tile.data = NULL;
  //}
  //if( ctx->C_tile.data != NULL){
  //  free(ctx->C_tile.data);
  //  ctx->C_tile.data = NULL;
  //}

  return QMCKL_SUCCESS;
}


qmckl_exit_code qmckl_dgemm_tiled_NN(qmckl_context context, int64_t Min, int64_t Nin, int64_t Kin,
				     double *A, int64_t incRowA,
				     double *B, int64_t incRowB,
				     double *C, int64_t incRowC) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_packed_matrix const packed_matrix_A;
  qmckl_packed_matrix const packed_matrix_B;
  qmckl_packed_matrix const packed_matrix_C;

  // Init memory
  qmckl_init_pack(context, packed_matrix_A, 'A', Min, Nin, Kin);
  qmckl_init_pack(context, packed_matrix_B, 'B', Min, Nin, Kin);
  qmckl_init_pack(context, packed_matrix_C, 'C', Min, Nin, Kin);

  // Tile A and B
  qmckl_pack_matrix(context, packed_matrix_A, 'A', Min, Kin, A, incRowA);
  qmckl_pack_matrix(context, packed_matrix_B, 'B', Kin, Nin, B, incRowB);
  qmckl_pack_matrix(context, packed_matrix_C, 'C', Min, Nin, C, incRowB);



  // Call DGEMM kernel
  qmckl_dgemm_tiled_avx2_nn(context, packed_matrix_A, incRowA,
             packed_matrix_B, incRowB,
             packed_matrix_C, incRowC);

  // Unpacking
  qmckl_unpack_matrix(context, packed_matrix_C, C, Min, Nin);

  //// Free memory
  qmckl_packed_matrix_destroy(packed_matrix_A);
  qmckl_packed_matrix_destroy(packed_matrix_B);
  qmckl_packed_matrix_destroy(packed_matrix_C);
  //qmckl_context_destroy(context);

  return QMCKL_SUCCESS;
}
