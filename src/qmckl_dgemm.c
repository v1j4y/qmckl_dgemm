#include "utils.h"
#include "qmckl_dgemm.h"
#include "qmckl_dgemm_private.h"

// Global context
context ctxt = {.qmckl_M=0, .qmckl_N=0, .qmckl_K=0, .MC=0, .NC=0, .KC=0, 
                       ._A_tile=NULL, ._B_tile=NULL, ._C_tile=NULL,
                       ._A=NULL, ._B=NULL};
context_p ctxtp = &ctxt;

int dgemm_main_tiled_avx2(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int64_t mb = Min / ctxtp[0].MC;
    int64_t nb = Nin / ctxtp[0].NC;
    int64_t kb = Kin / ctxtp[0].KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = ctxtp[0].MC*ctxtp[0].NC;
    size_t szeA = ctxtp[0].MC*ctxtp[0].KC*sizeof(double);
    size_t szeB = ctxtp[0].NC*ctxtp[0].KC*sizeof(double);
    int i,j,k,imb;
    int MCKC = ctxtp[0].MC*ctxtp[0].KC;
    int NCKC = ctxtp[0].NC*ctxtp[0].KC;

    int64_t i_tile_b, i_tile_a;
    double *A_tile_p __attribute__ ((aligned(64)));
    double *B_tile_p __attribute__ ((aligned(64)));
    double *C_tile_p __attribute__ ((aligned(64)));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    B_tile_p = ctxtp[0]._B_tile;
    A_tile_p = ctxtp[0]._A_tile;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = ctxtp[0]._A_tile;
        //B_tile_p = _B_tile + i*kb*(NCKC);
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = ctxtp[0].KC;

            idxi = i * ctxtp[0].NC * incRowC;
            idxk = k * ctxtp[0].KC * incColA;

            //B_tile_p = _B_tile + i_tile_b * (NCKC);
            C_tile_p = ctxtp[0]._C_tile + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_16regs(ctxtp[0].MC, ctxtp[0].KC, ctxtp[0].NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
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

    return 1;
}

//int dgemm_main_tiled_avx2_8regs(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
//                                                double *B, int64_t incRowB, int64_t incColB,
//                                                double *C, int64_t incRowC, int64_t incColC) {
//
//    int64_t mb = Min / MC;
//    int64_t nb = Nin / NC;
//    int64_t kb = Kin / KC;
//    int64_t idxi = 0;
//    int64_t idxk = 0;
//    int64_t nmbnb = 0;
//    int64_t nmbnb_prev = 0;
//    int64_t MCNC = MC*NC;
//    size_t szeA = MC*KC*sizeof(double);
//    size_t szeB = NC*KC*sizeof(double);
//    int i,j,k,imb;
//    int MCKC = MC*KC;
//
//    int64_t i_tile_b, i_tile_a;
//    double *A_tile_p __attribute__ ((aligned(64)));
//    double *B_tile_p __attribute__ ((aligned(64)));
//    double *C_tile_p __attribute__ ((aligned(64)));
//
//    // Initialize indices
//    i_tile_b = 0;
//    i_tile_a = 0;
//
////#pragma omp parallel
////{
//    for(i=0;i<nb;++i) {
//        nmbnb_prev = i*mb;
//        i_tile_a = 0;
//        A_tile_p = _A_tile;
//        imb = i*mb;
////#pragma omp single
//        for(k=0;k<kb;++k) {
//            int64_t kc = KC;
//
//            idxi = i * NC * incRowC;
//            idxk = k * KC * incColA;
//
//            B_tile_p = _B_tile + i_tile_b * (NC*KC);
//            C_tile_p = C + (imb) * MCNC;
//
//            for(j=0;j<mb;++j) {
//                #pragma forceinline
//                dgemm_macro_kernel_avx2_8regs(MC, KC, NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
//                A_tile_p += (MCKC);
//                C_tile_p +=  MCNC;
//                //nmbnb = nmbnb + 1;
//                i_tile_a += 1;
//            }
//            //if(k < (kb-1)) nmbnb = nmbnb_prev;
//            i_tile_b += 1;
//        }
//    }
////}
//
//    return 1;
//}


//int dgemm_main_tiled_sse2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
//                                                double *B, int64_t incRowB, int64_t incColB,
//                                                double *C, int64_t incRowC, int64_t incColC) {
//
//    int64_t mb = Min / MC;
//    int64_t nb = Nin / NC;
//    int64_t kb = Kin / KC;
//    int64_t idxi = 0;
//    int64_t idxk = 0;
//    int64_t nmbnb = 0;
//    int64_t nmbnb_prev = 0;
//    int64_t MCNC = MC*NC;
//    size_t szeA = MC*KC*sizeof(double);
//    size_t szeB = NC*KC*sizeof(double);
//    int i,j,k,imb;
//    int MCKC = MC*KC;
//
//    int64_t i_tile_b, i_tile_a;
//    double *A_tile_p __attribute__ ((aligned(64)));
//    double *B_tile_p __attribute__ ((aligned(64)));
//    double *C_tile_p __attribute__ ((aligned(64)));
//
//    // Initialize indices
//    i_tile_b = 0;
//    i_tile_a = 0;
//
////#pragma omp parallel
////{
//    for(i=0;i<nb;++i) {
//        nmbnb_prev = i*mb;
//        i_tile_a = 0;
//        A_tile_p = _A_tile;
//        imb = i*mb;
////#pragma omp single
//        for(k=0;k<kb;++k) {
//            int64_t kc = KC;
//
//            idxi = i * NC * incRowC;
//            idxk = k * KC * incColA;
//
//            B_tile_p = _B_tile + i_tile_b * (NC*KC);
//            C_tile_p = C + (imb) * MCNC;
//
//            for(j=0;j<mb;++j) {
//                #pragma forceinline
//                dgemm_macro_kernel_sse2_8regs(MC, KC, NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
//                A_tile_p += (MCKC);
//                C_tile_p +=  MCNC;
//                //nmbnb = nmbnb + 1;
//                i_tile_a += 1;
//            }
//            //if(k < (kb-1)) nmbnb = nmbnb_prev;
//            i_tile_b += 1;
//        }
//    }
////}
//
//    return 1;
//}

void unpackC(context_p ctxtp, double *B, int64_t M, int64_t N) {
    int64_t mb = M / ctxtp[0].MC;
    int64_t nb = N / ctxtp[0].NC;
    int64_t mp = ctxtp[0].MC / MR;
    int64_t np = ctxtp[0].NC / NR;
    int i,j;
    for(i=0;i<N;++i) {
        for(j=0;j<M;++j) {
          int64_t kmc = ( j / ctxtp[0].MC );
          int64_t lnc = ( i / ctxtp[0].NC );
          int64_t kmr = ( j - kmc * ctxtp[0].MC ) / MR;
          int64_t lnr = ( i - lnc * ctxtp[0].NC ) / NR;
          int64_t k   = ( ( j - kmc * ctxtp[0].MC ) - ( kmr * MR ) );
          int64_t l   = ( ( i - lnc * ctxtp[0].NC ) - ( lnr * NR ) );
          B[j + i*M] = ctxtp[0]._C_tile[(ctxtp[0].MC*ctxtp[0].NC)*(lnc*mb) + (ctxtp[0].MC*ctxtp[0].NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k];
        }
    }
}

//void init_dims_avx512() {
//
//    /*
//     * AVX512: We work only in factors of 16 * 14
//     */
//    //qmckl_M = (int64_t *)malloc(1 * sizeof(int64_t));
//    //qmckl_N = (int64_t *)malloc(1 * sizeof(int64_t));
//    //qmckl_K = (int64_t *)malloc(1 * sizeof(int64_t));
//
//    if((MAT_DIM_M % MC) != 0){
//
//      qmckl_M = (int64_t)((MAT_DIM_M/MC)+1)*MC;
//    }
//    else{
//      qmckl_M = (int64_t)MAT_DIM_M;
//    }
//
//    if((MAT_DIM_K % KC) != 0){
//      qmckl_K = (int64_t)((MAT_DIM_K/KC)+1)*KC;
//    }
//    else{
//      qmckl_K = (int64_t)MAT_DIM_K;
//    }
//
//    if((MAT_DIM_N % NR) != 0){
//      qmckl_N = (int64_t)((MAT_DIM_N/NR)+1)*NR;
//    }
//    else{
//      qmckl_N = (int64_t)MAT_DIM_N;
//    }
//    printf("M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)qmckl_M,(long)qmckl_K,(long)qmckl_N,(long)MC,(long)KC,(long)NC);
//}
//
//void init_dims_avx2() {
//
//    /*
//     * AVX2: We work only in factors of 8 * 6
//     */
//    //qmckl_M = (int64_t *)malloc(1 * sizeof(int64_t));
//    //qmckl_N = (int64_t *)malloc(1 * sizeof(int64_t));
//    //qmckl_K = (int64_t *)malloc(1 * sizeof(int64_t));
//    //
//    int MR2NR2 = MR2*NR2;
//
//    KC = MAT_DIM_K;
//
//    if((MAT_DIM_M % MR2) != 0){
//
//      qmckl_M = (int64_t)((MAT_DIM_M/MR2)+1)*MR2;
//      MC = qmckl_M;
//      if(qmckl_M > 1152) MC = qmckl_M/2;
//    }
//    else{
//      qmckl_M = (int64_t)MAT_DIM_M;
//      MC = qmckl_M;
//      if(qmckl_M > 1152) MC = qmckl_M/2;
//    }
//
//    if((MAT_DIM_K % KC) != 0){
//      qmckl_K = (int64_t)((MAT_DIM_K/KC)+1)*KC;
//    }
//    else{
//      qmckl_K = (int64_t)MAT_DIM_K;
//    }
//
//    if((MAT_DIM_N % NR2) != 0){
//      qmckl_N = (int64_t)((MAT_DIM_N/NR2)+1)*NR2;
//      NC = qmckl_N;
//      if(qmckl_N > 1152) NC = qmckl_N/2;
//    }
//    else{
//      qmckl_N = (int64_t)MAT_DIM_N;
//      NC = qmckl_N;
//      if(qmckl_N > 1152) NC = qmckl_N/2;
//    }
//    printf("(AVX2) M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)qmckl_M,(long)qmckl_K,(long)qmckl_N,(long)MC,(long)KC,(long)NC);
//}

void init_dims_avx2_input(context_p ctxtp, int64_t DIM_M, int64_t DIM_N, int64_t DIM_K) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
    //qmckl_M = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_N = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_K = (int64_t *)malloc(1 * sizeof(int64_t));
    //
    int MR2NR2 = MR2*NR2;

    ctxtp[0].KC = DIM_K;

    if((DIM_M % MR2) != 0){

      ctxtp[0].qmckl_M = (int64_t)((DIM_M/MR2)+1)*MR2;
      ctxtp[0].MC = ctxtp[0].qmckl_M;
      if(ctxtp[0].qmckl_M > 1152) ctxtp[0].MC = ctxtp[0].qmckl_M/2;
    }
    else{
      ctxtp[0].qmckl_M = (int64_t)DIM_M;
      ctxtp[0].MC = ctxtp[0].qmckl_M;
      if(ctxtp[0].qmckl_M > 1152) ctxtp[0].MC = ctxtp[0].qmckl_M/2;
    }

    if((DIM_K % ctxtp[0].KC) != 0){
      ctxtp[0].qmckl_K = (int64_t)((DIM_K/ctxtp[0].KC)+1)*ctxtp[0].KC;
    }
    else{
      ctxtp[0].qmckl_K = (int64_t)DIM_K;
    }

    if((DIM_N % NR2) != 0){
      ctxtp[0].qmckl_N = (int64_t)((DIM_N/NR2)+1)*NR2;
      ctxtp[0].NC = ctxtp[0].qmckl_N;
      if(ctxtp[0].qmckl_N > 1152) ctxtp[0].NC = ctxtp[0].qmckl_N/2;
    }
    else{
      ctxtp[0].qmckl_N = (int64_t)DIM_N;
      ctxtp[0].NC = ctxtp[0].qmckl_N;
      if(ctxtp[0].qmckl_N > 1152) ctxtp[0].NC = ctxtp[0].qmckl_N/2;
    }
    printf("(AVX2) M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)ctxtp[0].qmckl_M,(long)ctxtp[0].qmckl_K,(long)ctxtp[0].qmckl_N,(long)ctxtp[0].MC,(long)ctxtp[0].KC,(long)ctxtp[0].NC);
}


//int tile_matrix(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
//                                                double *B, int64_t incRowB, int64_t incColB,
//                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {
//
//    int64_t mb = qmckl_M / MC;
//    int64_t nb = qmckl_N / NC;
//    int64_t kb = qmckl_K / KC;
//    int64_t idxi = 0;
//    int64_t idxk = 0;
//    int64_t nmbnb = 0;
//    int64_t nmbnb_prev = 0;
//    int64_t MCNC = MC*NC;
//    int64_t NCKC = NC*KC;
//    int64_t i_tile_b, i_tile_a;
//    int i,j,k,ii;
//
////static double _A[MC*KC] __attribute__ ((aligned(64)));
////static double _B[NC*KC] __attribute__ ((aligned(64)));
//    if( _A == NULL) _A  = (double *)aligned_alloc(64, MC*KC * sizeof(double));
//    if( _B == NULL) _B  = (double *)aligned_alloc(64, NC*KC * sizeof(double));
//    if(_A_tile == NULL) _A_tile = (double *)aligned_alloc(64, qmckl_M*qmckl_K*2 * sizeof(double));
//    if(_B_tile == NULL) _B_tile = (double *)aligned_alloc(64, qmckl_N*qmckl_K*2 * sizeof(double));
//
//    // Initialize indices
//    i_tile_b = 0;
//    i_tile_a = 0;
//
//    for(i=0;i<nb;++i) {
//        for(k=0;k<kb;++k) {
//            int64_t kc = KC;
//            packB(kc, &B[k*KC*incRowB + i*NC*incColB], incRowB, incColB, _B);
//
//            // Write to tiled matrix to B
//            for(ii=0;ii<NCKC;++ii) {
//              _B_tile[i_tile_b * (NCKC) + ii] = _B[ii];
//            }
//            i_tile_b += 1;
//        }
//    }
//
//    for(k=0;k<kb;++k) {
//        int64_t kc = KC;
//
//        idxk = k * KC * incColA;
//
//        for(j=0;j<mb;++j) {
//            packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);
//
//            // Write to tiled matrix to A
//            for(ii=0;ii<MC*KC;++ii) {
//              _A_tile[i_tile_a * (MC*KC) + ii] = _A[ii];
//            }
//            i_tile_a += 1;
//        }
//    }
//
//    return 1;
//}

int tile_matrix_general(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {

    int64_t mb = ctxtp[0].qmckl_M / ctxtp[0].MC;
    int64_t nb = ctxtp[0].qmckl_N / ctxtp[0].NC;
    int64_t kb = ctxtp[0].qmckl_K / ctxtp[0].KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = ctxtp[0].MC*ctxtp[0].NC;
    int64_t NCKC = ctxtp[0].NC*ctxtp[0].KC;
    int64_t MCKC = ctxtp[0].MC*ctxtp[0].KC;
    int64_t MCmax, NCmax;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;

//static double _A[MC*KC] __attribute__ ((aligned(64)));
//static double _B[NC*KC] __attribute__ ((aligned(64)));
    if( ctxtp[0]._A == NULL) ctxtp[0]._A  = (double *)aligned_alloc(64, ctxtp[0].MC*ctxtp[0].KC * sizeof(double));
    if( ctxtp[0]._B == NULL) ctxtp[0]._B  = (double *)aligned_alloc(64, ctxtp[0].NC*ctxtp[0].KC * sizeof(double));
    if( ctxtp[0]._A_tile == NULL) ctxtp[0]._A_tile = (double *)aligned_alloc(64, ctxtp[0].qmckl_M*ctxtp[0].qmckl_K*2 * sizeof(double));
    if( ctxtp[0]._B_tile == NULL) ctxtp[0]._B_tile = (double *)aligned_alloc(64, ctxtp[0].qmckl_N*ctxtp[0].qmckl_K*2 * sizeof(double));
    if( ctxtp[0]._B_tile == NULL) ctxtp[0]._B_tile = (double *)aligned_alloc(64, ctxtp[0].qmckl_N*ctxtp[0].qmckl_K*2 * sizeof(double));
    if( ctxtp[0]._C_tile == NULL) ctxtp[0]._C_tile = (double *)aligned_alloc(64, ctxtp[0].qmckl_M*ctxtp[0].qmckl_N   * sizeof(double));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    for(i=0;i<nb;++i) {
        if ( (i+1)*ctxtp[0].NC > Nin){
          NCmax = Nin - (i+0)*ctxtp[0].NC;
        }
        else{
          NCmax = ctxtp[0].NC;
        }
        for(k=0;k<kb;++k) {
            int64_t kc = ctxtp[0].KC;
            packB_general(ctxtp, kc, NCmax, &B[k*ctxtp[0].KC*incRowB + i*ctxtp[0].NC*incColB], incRowB, incColB, ctxtp[0]._B);

            // Write to tiled matrix to B
            for(ii=0;ii<NCKC;++ii) {
              ctxtp[0]._B_tile[i_tile_b * (NCKC) + ii] = ctxtp[0]._B[ii];
            }
            i_tile_b += 1;
        }
    }

    for(k=0;k<kb;++k) {
        int64_t kc = ctxtp[0].KC;

        idxk = k * ctxtp[0].KC * incColA;

        for(j=0;j<mb;++j) {
            if ( (j+1)*ctxtp[0].MC > Min){
              MCmax = Min - (j+0)*ctxtp[0].MC;
            }
            else{
              MCmax = ctxtp[0].MC;
            }
            packA_general(ctxtp, kc, MCmax, &A[idxk + j*ctxtp[0].MC*incRowA], incRowA, incColA, ctxtp[0]._A);

            // Write to tiled matrix to A
            for(ii=0;ii<MCKC;++ii) {
              ctxtp[0]._A_tile[i_tile_a * (MCKC) + ii] = ctxtp[0]._A[ii];
            }
            i_tile_a += 1;
        }
    }

    return 1;
}

void free_context(context_p ctxtp){

    if( ctxtp[0]._A != NULL){
      free(ctxtp[0]._A);
      ctxtp[0]._A = NULL;
    }
    if( ctxtp[0]._B != NULL){
      free(ctxtp[0]._B);
      ctxtp[0]._B = NULL;
    }
    if( ctxtp[0]._A_tile != NULL){
      free(ctxtp[0]._A_tile);
      ctxtp[0]._A_tile = NULL;
    }
    if( ctxtp[0]._B_tile != NULL){
      free(ctxtp[0]._B_tile);
      ctxtp[0]._B_tile = NULL;
    }
    if( ctxtp[0]._B_tile != NULL){
      free(ctxtp[0]._B_tile);
      ctxtp[0]._B_tile = NULL;
    }
    if( ctxtp[0]._C_tile != NULL){
      free(ctxtp[0]._C_tile);
      ctxtp[0]._C_tile = NULL;
    }
}


int dgemm_main_tiled(context_p ctxtp, int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {
    // Init memory
    init_dims_avx2_input(ctxtp, Min, Nin, Kin);

    // Tile A and B
    tile_matrix_general(ctxtp, Min, Nin, Kin, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC, ctxtp[0]._A_tile, ctxtp[0]._B_tile);


    // Call DGEMM kernel
    dgemm_main_tiled_avx2(ctxtp, ctxtp[0].qmckl_M, ctxtp[0].qmckl_N, ctxtp[0].qmckl_K, A, incRowA, incColA,
               B, incRowB, incColB,
               ctxtp[0]._C_tile, incRowC, incColC);

    // Unpacking
    unpackC(ctxtp, C, Min, Nin);

    // Free memory
    free(ctxtp);

    return 1;
}

int dgemm_naive(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int i,j,k;
    for(i=0;i<Min;++i) {
        for(j=0;j<Nin;++j) {
            for(k=0;k<Kin;++k) {
                C[i*Nin + j] = C[i*Nin + j] + A[i*Kin + k]*B[k*Nin + j];
            }
        }
    }

    return 1;
}

