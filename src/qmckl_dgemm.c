#include "utils.h"
#include "qmckl_dgemm.h"

int dgemm_main_tiled_avx2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int64_t mb = Min / MC;
    int64_t nb = Nin / NC;
    int64_t kb = Kin / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*NC;
    size_t szeA = MC*KC*sizeof(double);
    size_t szeB = NC*KC*sizeof(double);
    int i,j,k,imb;
    int MCKC = MC*KC;
    int NCKC = NC*KC;

    int64_t i_tile_b, i_tile_a;
    double *A_tile_p __attribute__ ((aligned(64)));
    double *B_tile_p __attribute__ ((aligned(64)));
    double *C_tile_p __attribute__ ((aligned(64)));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    B_tile_p = _B_tile;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = _A_tile;
        //B_tile_p = _B_tile + i*kb*(NCKC);
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = KC;

            idxi = i * NC * incRowC;
            idxk = k * KC * incColA;

            //B_tile_p = _B_tile + i_tile_b * (NCKC);
            C_tile_p = C + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_16regs(MC, KC, NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
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

int dgemm_main_tiled_avx2_8regs(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int64_t mb = Min / MC;
    int64_t nb = Nin / NC;
    int64_t kb = Kin / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*NC;
    size_t szeA = MC*KC*sizeof(double);
    size_t szeB = NC*KC*sizeof(double);
    int i,j,k,imb;
    int MCKC = MC*KC;

    int64_t i_tile_b, i_tile_a;
    double *A_tile_p __attribute__ ((aligned(64)));
    double *B_tile_p __attribute__ ((aligned(64)));
    double *C_tile_p __attribute__ ((aligned(64)));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = _A_tile;
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = KC;

            idxi = i * NC * incRowC;
            idxk = k * KC * incColA;

            B_tile_p = _B_tile + i_tile_b * (NC*KC);
            C_tile_p = C + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_avx2_8regs(MC, KC, NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
                A_tile_p += (MCKC);
                C_tile_p +=  MCNC;
                //nmbnb = nmbnb + 1;
                i_tile_a += 1;
            }
            //if(k < (kb-1)) nmbnb = nmbnb_prev;
            i_tile_b += 1;
        }
    }
//}

    return 1;
}


int dgemm_main_tiled_sse2(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {

    int64_t mb = Min / MC;
    int64_t nb = Nin / NC;
    int64_t kb = Kin / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*NC;
    size_t szeA = MC*KC*sizeof(double);
    size_t szeB = NC*KC*sizeof(double);
    int i,j,k,imb;
    int MCKC = MC*KC;

    int64_t i_tile_b, i_tile_a;
    double *A_tile_p __attribute__ ((aligned(64)));
    double *B_tile_p __attribute__ ((aligned(64)));
    double *C_tile_p __attribute__ ((aligned(64)));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

//#pragma omp parallel
//{
    for(i=0;i<nb;++i) {
        nmbnb_prev = i*mb;
        i_tile_a = 0;
        A_tile_p = _A_tile;
        imb = i*mb;
//#pragma omp single
        for(k=0;k<kb;++k) {
            int64_t kc = KC;

            idxi = i * NC * incRowC;
            idxk = k * KC * incColA;

            B_tile_p = _B_tile + i_tile_b * (NC*KC);
            C_tile_p = C + (imb) * MCNC;

            for(j=0;j<mb;++j) {
                #pragma forceinline
                dgemm_macro_kernel_sse2_8regs(MC, KC, NC, C_tile_p, incRowC, incColC, A_tile_p, B_tile_p);
                A_tile_p += (MCKC);
                C_tile_p +=  MCNC;
                //nmbnb = nmbnb + 1;
                i_tile_a += 1;
            }
            //if(k < (kb-1)) nmbnb = nmbnb_prev;
            i_tile_b += 1;
        }
    }
//}

    return 1;
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

void init_dims_avx2_input(int64_t DIM_M, int64_t DIM_N, int64_t DIM_K) {

    /*
     * AVX2: We work only in factors of 8 * 6
     */
    //qmckl_M = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_N = (int64_t *)malloc(1 * sizeof(int64_t));
    //qmckl_K = (int64_t *)malloc(1 * sizeof(int64_t));
    //
    int MR2NR2 = MR2*NR2;

    KC = DIM_K;

    if((DIM_M % MR2) != 0){

      qmckl_M = (int64_t)((DIM_M/MR2)+1)*MR2;
      MC = qmckl_M;
      if(qmckl_M > 1152) MC = qmckl_M/2;
    }
    else{
      qmckl_M = (int64_t)DIM_M;
      MC = qmckl_M;
      if(qmckl_M > 1152) MC = qmckl_M/2;
    }

    if((DIM_K % KC) != 0){
      qmckl_K = (int64_t)((DIM_K/KC)+1)*KC;
    }
    else{
      qmckl_K = (int64_t)DIM_K;
    }

    if((DIM_N % NR2) != 0){
      qmckl_N = (int64_t)((DIM_N/NR2)+1)*NR2;
      NC = qmckl_N;
      if(qmckl_N > 1152) NC = qmckl_N/2;
    }
    else{
      qmckl_N = (int64_t)DIM_N;
      NC = qmckl_N;
      if(qmckl_N > 1152) NC = qmckl_N/2;
    }
    printf("(AVX2) M=%ld K=%ld N=%ld | MC=%ld KC=%ld NC=%ld\n",(long)qmckl_M,(long)qmckl_K,(long)qmckl_N,(long)MC,(long)KC,(long)NC);
}


int tile_matrix(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {

    int64_t mb = qmckl_M / MC;
    int64_t nb = qmckl_N / NC;
    int64_t kb = qmckl_K / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*NC;
    int64_t NCKC = NC*KC;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;

//static double _A[MC*KC] __attribute__ ((aligned(64)));
//static double _B[NC*KC] __attribute__ ((aligned(64)));
    if( _A == NULL) _A  = (double *)aligned_alloc(64, MC*KC * sizeof(double));
    if( _B == NULL) _B  = (double *)aligned_alloc(64, NC*KC * sizeof(double));
    if(_A_tile == NULL) _A_tile = (double *)aligned_alloc(64, qmckl_M*qmckl_K*2 * sizeof(double));
    if(_B_tile == NULL) _B_tile = (double *)aligned_alloc(64, qmckl_N*qmckl_K*2 * sizeof(double));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    for(i=0;i<nb;++i) {
        for(k=0;k<kb;++k) {
            int64_t kc = KC;
            packB(kc, &B[k*KC*incRowB + i*NC*incColB], incRowB, incColB, _B);

            // Write to tiled matrix to B
            for(ii=0;ii<NCKC;++ii) {
              _B_tile[i_tile_b * (NCKC) + ii] = _B[ii];
            }
            i_tile_b += 1;
        }
    }

    for(k=0;k<kb;++k) {
        int64_t kc = KC;

        idxk = k * KC * incColA;

        for(j=0;j<mb;++j) {
            packA(kc, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

            // Write to tiled matrix to A
            for(ii=0;ii<MC*KC;++ii) {
              _A_tile[i_tile_a * (MC*KC) + ii] = _A[ii];
            }
            i_tile_a += 1;
        }
    }

    return 1;
}

int tile_matrix_general(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC, double *A_tile, double *B_tile) {

    int64_t mb = qmckl_M / MC;
    int64_t nb = qmckl_N / NC;
    int64_t kb = qmckl_K / KC;
    int64_t idxi = 0;
    int64_t idxk = 0;
    int64_t nmbnb = 0;
    int64_t nmbnb_prev = 0;
    int64_t MCNC = MC*NC;
    int64_t NCKC = NC*KC;
    int64_t MCmax, NCmax;
    int64_t i_tile_b, i_tile_a;
    int i,j,k,ii;

//static double _A[MC*KC] __attribute__ ((aligned(64)));
//static double _B[NC*KC] __attribute__ ((aligned(64)));
    if( _A == NULL) _A  = (double *)aligned_alloc(64, MC*KC * sizeof(double));
    if( _B == NULL) _B  = (double *)aligned_alloc(64, NC*KC * sizeof(double));
    if(_A_tile == NULL) _A_tile = (double *)aligned_alloc(64, qmckl_M*qmckl_K*2 * sizeof(double));
    if(_B_tile == NULL) _B_tile = (double *)aligned_alloc(64, qmckl_N*qmckl_K*2 * sizeof(double));

    // Initialize indices
    i_tile_b = 0;
    i_tile_a = 0;

    for(i=0;i<nb;++i) {
        if ( (i+1)*NC > Nin){
          NCmax = Nin - (i+0)*NC;
        }
        else{
          NCmax = NC;
        }
        for(k=0;k<kb;++k) {
            int64_t kc = KC;
            packB_general(kc, NCmax, &B[k*KC*incRowB + i*NC*incColB], incRowB, incColB, _B);

            // Write to tiled matrix to B
            for(ii=0;ii<NCKC;++ii) {
              _B_tile[i_tile_b * (NCKC) + ii] = _B[ii];
            }
            i_tile_b += 1;
        }
    }

    for(k=0;k<kb;++k) {
        int64_t kc = KC;

        idxk = k * KC * incColA;

        for(j=0;j<mb;++j) {
            if ( (j+1)*MC > Min){
              MCmax = Min - (j+0)*MC;
            }
            else{
              MCmax = MC;
            }
            packA_general(kc, MCmax, &A[idxk + j*MC*incRowA], incRowA, incColA, _A);

            // Write to tiled matrix to A
            for(ii=0;ii<MC*KC;++ii) {
              _A_tile[i_tile_a * (MC*KC) + ii] = _A[ii];
            }
            i_tile_a += 1;
        }
    }

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


int dgemm_main_tiled(int64_t Min, int64_t Nin, int64_t Kin, double *A, int64_t incRowA, int64_t incColA,
                                                double *B, int64_t incRowB, int64_t incColB,
                                                double *C, int64_t incRowC, int64_t incColC) {
    init_dims_avx2_input(Min, Nin, Kin);

    // Tile A and B
    tile_matrix_general(Min, Nin, Kin, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC, _A_tile, _B_tile);


    // Call DGEMM kernel
    dgemm_main_tiled_avx2(qmckl_M, qmckl_N, qmckl_K, A, incRowA, incColA,
               B, incRowB, incColB,
               C, incRowC, incColC);
    return 1;
}

