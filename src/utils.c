#include "utils.h"

unsigned long long rdtsc(void)
{
  unsigned long long a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}

void fill_matrix_ones(double *A, int64_t dim) {
  int i;
    for(i=0;i<dim;++i) {
        A[i] = 1.0;
    }
}

void fill_matrix_random(double *dA, int64_t dM, int64_t dN) {
    double count=0.0;
    int i,j;
    //srand ( 1024 );
    for(i=0;i<dM;++i) {
      for(j=0;j<dN;++j) {
        double random_value = (double)rand()/RAND_MAX*2.0-1.0;//float in range -1 to 1
        dA[i*dN + j] = random_value;
      }
    }
}

void fill_matrix_uniform(double *dA, int64_t dM, int64_t dN) {
    double count=0.0;
    int i,j;
    for(i=0;i<dM;++i) {
      count = 0.0;
      for(j=0;j<dN;++j) {
        dA[i*dN + j] = count;
        count += 1.0;
      }
    }
}

void fill_matrix_uniform_2(double *dA, int64_t dM, int64_t dN) {
    double count=0.0;
    int i,j;
    for(i=0;i<dN;++i) {
      count = 0.0;
      for(j=0;j<dM;++j) {
        dA[j*dN + i] = count;
        count += 1.0;
      }
    }
}

void fill_matrix_zeros(double *A, int64_t dim) {
  int i;
    for(i=0;i<dim;++i) {
        A[i] = 0.0;
    }
}

void copy_matrix(double *toB, double *fromA, int64_t M, int64_t N) {
  int i,j;
    for(i=0;i<M;++i) {
        for(j=0;j<N;++j) {
            toB[i*N + j] = fromA[i*N + j];
        }
    }
}

void copy_matrix_T(double *toB, double *fromA, int64_t M, int64_t N) {
  int i,j;
    for(i=0;i<M;++i) {
        for(j=0;j<N;++j) {
            toB[j*M + i] = fromA[i*N + j];
        }
    }
}

void print_matrix(double *A, int64_t M, int64_t N) {
  int i,j;
    for(i=0;i<M;++i) {
        for(j=0;j<N;++j) {
            printf(" %5.3f ",A[i*N + j]);
        }
        printf("\n");
    }
}

void print_matrix_T(double *A, int64_t M, int64_t N) {
  int i,j;
    for(i=0;i<N;++i) {
        for(j=0;j<M;++j) {
            printf(" %5.3f ",A[i + j*N]);
        }
        printf("\n");
    }
}

void print_diff_matrix(double *A, double *B, int64_t M, int64_t N) {
  int i,j;
    for(j=0;j<N;++j) {
        for(i=0;i<M;++i) {
          if( abs(A[i + j*M] - B[i + j*M]) > 1e-13){
            printf(" %5.3f ",abs(A[i + j*M] - B[i + j*M]));
            printf("Fail\n");
            return;
          };
        }
    }
}

void print_diff_matrix_ABT(double *A, double *B, int64_t M, int64_t N) {
  int i,j, idi,idj, idtot;
  idtot=0;
    for(j=0;j<N;++j) {
        for(i=0;i<M;++i) {
	  idi = idtot % N;
	  idj = (idtot - idi)/N;
	  //printf(" %5.3f (%5.3f) ",A[i + j*M], B[idi*M + idj]);
	  //printf(" %5.3f ((%d,%d)) ",A[i + j*M], idi,idj);
	  if( abs(A[i + j*M] - B[idi*M + idj]) > 1e-13){
	    printf(" %5.3f ",abs(A[i + j*M] - B[idi*M + idj]));
	    printf("Fail\n");
	    return;
	  };
	  idtot+=1;
        }
    }
}

qmckl_exit_code get_diff_matrix(double *A, double *B, int64_t M, int64_t N) {
  int i,j;
    for(j=0;j<N;++j) {
        for(i=0;i<M;++i) {
          if( abs(A[i + j*M] - B[i + j*M]) > 1e-13){
            return QMCKL_FAILURE;
          };
        }
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code get_diff_matrix_ABT(double *A, double *B, int64_t M, int64_t N) {
  int i,j, idi,idj, idtot;
  idtot=0;
  for(j=0;j<N;++j) {
      for(i=0;i<M;++i) {
	idi = idtot % N;
	idj = (idtot - idi)/N;
	if( abs(A[i + j*M] - B[idi*M + idj]) > 1e-13){
          return QMCKL_FAILURE;
        };
	idtot+=1;
      }
  }
  return QMCKL_SUCCESS;
}

//void print_matrix_ASer(double *A, int64_t M, int64_t N) {
//    int64_t mb = M / MC;
//    int64_t nb = N / N;
//    int64_t mp = MC / MR;
//    int64_t np = NC / NR;
//    int64_t idx = 0;
//    int i,j;
//    for(i=0;i<N;++i) {
//        for(j=0;j<M;++j) {
//          int64_t kmc = ( j / MC );
//          int64_t lnc = ( i / NC );
//          int64_t kmr = ( j - kmc * MC ) / MR;
//          int64_t lnr = ( i - lnc * NC ) / NR;
//          int64_t k   = ( ( j - kmc * MC ) - ( kmr * MR ) );
//          int64_t l   = ( ( i - lnc * NC ) - ( lnr * NR ) );
//          //printf("\n (%d,%d) knc = %ld lmc = %ld\n",i,j, knc, lmc);
//          //printf(" knr = %ld lmr = %ld\n",knr, lmr);
//          //printf(" k   = %ld l   = %ld\n",k, l);
//          //printf(" id = %ld\n",(MC*NC)*(lnc*mb) + (MC*NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k);
//          printf(" %5.3f ",A[(MC*NC)*(lnc*mb) + (MC*NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k]);
//          //printf(" %5.3f ",A[idx]);
//          //idx = idx + 1;
//        }
//        printf("\n");
//    }
//}

//void print_diff_matrix_AT_B(double *A, double *B, int64_t M, int64_t N) {
//  int i,j;
//    for(j=0;j<N;++j) {
//        for(i=0;i<M;++i) {
//            printf(" %5.3f ",abs(A[j + i*N] - B[i + j*M]));
//        }
//        printf("\n");
//    }
//}

void print_diff_matrix_ASer_BT(qmckl_context context, qmckl_tile_matrix tile_matrix, double *A, double *B, int64_t M, int64_t N) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_tile_struct* const tmat = (qmckl_tile_struct* const) tile_matrix;
    int64_t mc = tmat->MCt;
    int64_t nc = tmat->NCt;
    int64_t mb = M / mc;
    int64_t nb = N / nc;
    int64_t mp = mc / MR;
    int64_t np = nc / NR;
    int64_t MCNC = mc*nc;
    int i,j;
    for(i=0;i<N;++i) {
        for(j=0;j<M;++j) {
          int64_t kmc = ( j / mc );
          int64_t lnc = ( i / nc );
          int64_t kmr = ( j - kmc * mc ) / MR;
          int64_t lnr = ( i - lnc * nc ) / NR;
          int64_t k   = ( ( j - kmc * mc ) - ( kmr * MR ) );
          int64_t l   = ( ( i - lnc * nc ) - ( lnr * NR ) );
          if(abs(A[(MCNC)*(lnc*mb) + (MCNC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k] - B[j + i*M]) > 1e-13){
            printf(" %20.10e ",abs(A[(MCNC)*(lnc*mb) + (MCNC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k] - B[j + i*M]));
            printf("Fail\n");
            return;
          };
        }
        //printf("\n");
    }
}

//void print_diff_matrix_ASer_B(double *A, double *B, int64_t M, int64_t N) {
//    int64_t mb = M / MC;
//    int64_t nb = N / NC;
//    int64_t mp = MC / MR;
//    int64_t np = NC / NR;
//    int i,j;
//    for(i=0;i<N;++i) {
//        for(j=0;j<M;++j) {
//          int64_t kmc = ( j / MC );
//          int64_t lnc = ( i / NC );
//          int64_t kmr = ( j - kmc * MC ) / MR;
//          int64_t lnr = ( i - lnc * NC ) / NR;
//          int64_t k   = ( ( j - kmc * MC ) - ( kmr * MR ) );
//          int64_t l   = ( ( i - lnc * NC ) - ( lnr * NR ) );
//          //printf(" %5.3f ",abs(A[(MC*NC)*(lnc*mb) + (MC*NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k] - B[i + j*N]));
//          if(abs(A[(MC*NC)*(lnc*mb) + (MC*NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k] - B[i + j*N]) > 1e-13){
//            printf(" %20.10e ",abs(A[(MC*NC)*(lnc*mb) + (MC*NC)*(kmc) + (MR*NR)*(lnr*mp) + (MR*NR)*(kmr) + (l*MR) + k] - B[i + j*N]));
//            printf("Fail\n");
//            return;
//          };
//        }
//        //printf("\n");
//    }
//}

// Pack A which is traversed row wise (MC rows and KC columns)
//void packA(int64_t kc, double *A, int64_t dimRowA, int64_t dimColA, double *buffer) {
//  int64_t mp = MC / MR;
//  double *buffer_start = buffer;
//  double *A_start = A;
//  int k,j,i;
//  for(k=0;k<mp;++k) {
//    for(j=0;j<MR;++j) {
//      for(i=0;i<kc;++i) {
//        buffer[j + i*MR] = A[j*dimRowA + i];
//      }
//    }
//    buffer = buffer + MR * kc;
//    A = A + MR * dimRowA;
//  }
//  //printf("\nPack A (%d %d)\n",dimRowA,dimColA);
//  //for(i=0;i<kc;++i){
//  //  for(j=0;j<MC;++j){
//  //    printf(" %5.3f ",buffer_start[i*MC + j]);
//  //  }
//  //  printf("\n");
//  //}
//  //exit(0);
//}
//
//// Pack B which is traversed column wise (NC rows and KC columns)
//void packB(int64_t kc, double *B, int64_t dimRowB, int64_t dimColB, double *buffer) {
//  int64_t np = NC / NR;
//  double *buffer_start = buffer;
//  double *B_start = B;
//  int k,j,i;
//  for(k=0;k<np;++k) {
//    for(i=0;i<kc;++i) {
//      for(j=0;j<NR;++j) {
//        buffer[j + i*NR] = B[j + i*dimRowB];
//      }
//    }
//    buffer = buffer + NR * kc;
//    B = B + NR;
//  }
//  //printf("\nPack B (%d %d)\n",dimRowB,dimColB);
//  //for(k=0;k<np;++k){
//  //for(i=0;i<kc;++i){
//  //  for(j=0;j<NR;++j){
//  //    printf(" %5.3f ",buffer_start[k*(NR*kc) + i*NR + j]);
//  //  }
//  //  printf("\n");
//  //}
//  //  printf("\n");
//  //  printf("\n");
//  //}
//  //exit(0);
//}

// Pack A which is traversed row wise (MC rows and KC columns)
void packA_general(qmckl_context context, qmckl_tile_matrix tile_matrix, int64_t kc, int64_t MCmax, double *A, int64_t dimRowA, int64_t dimColA, double *buffer) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_tile_struct* const tmat = (qmckl_tile_struct* const) tile_matrix;
  int64_t mp = tmat->MCt / MR;
  double *buffer_start = buffer;
  double *A_start = A;
  int k,j,i,idxMC;
  idxMC = 0;
  for(k=0;k<mp;++k) {
    for(j=0;j<MR;++j) {
      idxMC = k*MR + j;
      for(i=0;i<kc;++i) {
        if(idxMC < MCmax){
            buffer[j + i*MR] = A[j*dimRowA + i];
        }
        else{
            buffer[j + i*MR] = 0.0;
        }
      }
    }
    buffer = buffer + MR * kc;
    if(idxMC < MCmax) A = A + MR * dimRowA;
  }
}

// Pack B which is traversed column wise (NC rows and KC columns)
void packB_general(qmckl_context context, qmckl_tile_matrix tile_matrix, int64_t kc, int64_t NCmax, double *B, int64_t dimRowB, int64_t dimColB, double *buffer) {
  qmckl_context_struct* const ctx = (qmckl_context_struct* const) context;
  qmckl_tile_struct* const tmat = (qmckl_tile_struct* const) tile_matrix;
  int64_t np = tmat->NCt / NR;
  double *buffer_start = buffer;
  double *B_start = B;
  int k,j,i, idxNC, idxNCprev;
  idxNC = 0;
  for(k=0;k<np;++k) {
    for(i=0;i<kc;++i) {
      for(j=0;j<NR;++j) {
        idxNC = k*NR + j;
        if(idxNC < NCmax){
          buffer[j + i*NR] = B[j + i*dimRowB];
        }
        else{
          buffer[j + i*NR] = 0.0;
        }
      }
        idxNC = k*NR + 0;
    }
    buffer = buffer + NR * kc;
    if( idxNC < NCmax) B = B + NR;
  }
}

//void packA(int64_t kc, double *A, int64_t incRowA, int64_t incColA, double *buffer) {
//    int64_t mp = MC / MR;
//    double *buffer_start = buffer;
//    double *A_start = A;
//    printf("PackA \n");
//    printf("(%d %d) mp=%d\n",incRowA, incColA,mp);
//    const uint64_t t0 = rdtsc();
//    for(k=0;k<mp;++k) {
//        for(i=0;i<(kc - 0);i = i + 1) {
//            //for(j=0;j<(MR - 4);j = j + 4) {
//                buffer[0] = A[0*incRowA];
//                buffer[1] = A[1*incRowA];
//                buffer[2] = A[2*incRowA];
//                buffer[3] = A[3*incRowA];
//                buffer[4] = A[4*incRowA];
//                buffer[5] = A[5*incRowA];
//                buffer[6] = A[6*incRowA];
//                buffer[7] = A[7*incRowA];
//            //}
//            A = A + incColA; // incColA == 1
//            buffer = buffer + MR;
//        }
//        //buffer = buffer_start + MR * kc;
//        A = A_start + MR * incRowA;
//    }
//    const uint64_t dt = rdtsc() - t0;
//    printf("PackA = %lf\n", 1.0e-1 * dt);
//    exit(0);
//}
//
//void packB(int64_t kc, double *B, int64_t incRowB, int64_t incColB, double *buffer) {
//    int64_t np = NC / NR;
//    double *buffer_start = buffer;
//    double *B_start = B;
//    printf("PackB \n");
//    printf("(%d %d) np=%d\n",incRowB, incColB,np);
//    const uint64_t t0 = rdtsc();
//    //incRowB = incColB;
//    //incColB = 1;
//    for(k=0;k<np;++k) {
//        for(i=0;i<(kc - 0);i = i + 1) {
//                buffer[0] = B[0]; // incColB == 1
//                buffer[1] = B[1]; // incColB == 1
//                buffer[2] = B[2]; // incColB == 1
//                buffer[3] = B[3]; // incColB == 1
//                buffer[4] = B[4]; // incColB == 1
//                buffer[5] = B[5]; // incColB == 1
//                buffer[6] = B[6]; // incColB == 1
//                buffer[7] = B[7]; // incColB == 1
//            //}
//        //        buffer[8 ] = B[0*incColB + 1]; // incColB == 1
//        //        buffer[9 ] = B[1*incColB + 1]; // incColB == 1
//        //        buffer[10] = B[2*incColB + 1]; // incColB == 1
//        //        buffer[11] = B[3*incColB + 1]; // incColB == 1
//        //        buffer[12] = B[4*incColB + 1]; // incColB == 1
//        //        buffer[13] = B[5*incColB + 1]; // incColB == 1
//        //        buffer[14] = B[6*incColB + 1]; // incColB == 1
//        //        buffer[15] = B[7*incColB + 1]; // incColB == 1
//        //    //for(j=0;j<(NR - 4);j = j + 4) {
//        //        buffer[16] = B[0*incColB + 2]; // incColB == 1
//        //        buffer[17] = B[1*incColB + 2]; // incColB == 1
//        //        buffer[18] = B[2*incColB + 2]; // incColB == 1
//        //        buffer[19] = B[3*incColB + 2]; // incColB == 1
//        //        buffer[20] = B[4*incColB + 2]; // incColB == 1
//        //        buffer[21] = B[5*incColB + 2]; // incColB == 1
//        //        buffer[22] = B[6*incColB + 2]; // incColB == 1
//        //        buffer[23] = B[7*incColB + 2]; // incColB == 1
//        //    //}
//        //        buffer[24] = B[0*incColB + 3]; // incColB == 1
//        //        buffer[25] = B[1*incColB + 3]; // incColB == 1
//        //        buffer[26] = B[2*incColB + 3]; // incColB == 1
//        //        buffer[27] = B[3*incColB + 3]; // incColB == 1
//        //        buffer[28] = B[4*incColB + 3]; // incColB == 1
//        //        buffer[29] = B[5*incColB + 3]; // incColB == 1
//        //        buffer[30] = B[6*incColB + 3]; // incColB == 1
//        //        buffer[31] = B[7*incColB + 3]; // incColB == 1
//        //  //
//        //        buffer[32] = B[0*incColB + 4]; // incColB == 1
//        //        buffer[33] = B[1*incColB + 4]; // incColB == 1
//        //        buffer[34] = B[2*incColB + 4]; // incColB == 1
//        //        buffer[35] = B[3*incColB + 4]; // incColB == 1
//        //        buffer[36] = B[4*incColB + 4]; // incColB == 1
//        //        buffer[37] = B[5*incColB + 4]; // incColB == 1
//        //        buffer[38] = B[6*incColB + 4]; // incColB == 1
//        //        buffer[39] = B[7*incColB + 4]; // incColB == 1
//        //    //}
//        //        buffer[40] = B[0*incColB + 5]; // incColB == 1
//        //        buffer[41] = B[1*incColB + 5]; // incColB == 1
//        //        buffer[42] = B[2*incColB + 5]; // incColB == 1
//        //        buffer[43] = B[3*incColB + 5]; // incColB == 1
//        //        buffer[44] = B[4*incColB + 5]; // incColB == 1
//        //        buffer[45] = B[5*incColB + 5]; // incColB == 1
//        //        buffer[46] = B[6*incColB + 5]; // incColB == 1
//        //        buffer[47] = B[7*incColB + 5]; // incColB == 1
//        //    //for(j=0;j<(NR - 4);j = j + 4) {
//        //        buffer[48] = B[0*incColB + 6]; // incColB == 1
//        //        buffer[49] = B[1*incColB + 6]; // incColB == 1
//        //        buffer[50] = B[2*incColB + 6]; // incColB == 1
//        //        buffer[51] = B[3*incColB + 6]; // incColB == 1
//        //        buffer[52] = B[4*incColB + 6]; // incColB == 1
//        //        buffer[53] = B[5*incColB + 6]; // incColB == 1
//        //        buffer[54] = B[6*incColB + 6]; // incColB == 1
//        //        buffer[55] = B[7*incColB + 6]; // incColB == 1
//        //    //}
//        //        buffer[56] = B[0*incColB + 7]; // incColB == 1
//        //        buffer[57] = B[1*incColB + 7]; // incColB == 1
//        //        buffer[58] = B[2*incColB + 7]; // incColB == 1
//        //        buffer[59] = B[3*incColB + 7]; // incColB == 1
//        //        buffer[60] = B[4*incColB + 7]; // incColB == 1
//        //        buffer[61] = B[5*incColB + 7]; // incColB == 1
//        //        buffer[62] = B[6*incColB + 7]; // incColB == 1
//        //        buffer[63] = B[7*incColB + 7]; // incColB == 1
//            B = B + incRowB + 0 + 0;
//            buffer = buffer + NR; //+ NR + NR + NR + NR + NR + NR + NR;
//        }
//        //buffer = buffer_start + NR * kc;
//        B = B_start + NR * incColB;
//    }
//    const uint64_t dt = rdtsc() - t0;
//    printf("PackB = %f\n", 1e-0 * dt);
//}

