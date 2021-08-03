#include "kernel.h"
#include "utils.h"

void dgemm_kernel(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR];

    for(int l=0;l<MR*NR;++l) {
        AB[l] = 0.0;
    }

    for(int k=0;k<kc;++k) {
        for(int j=0;j<NR;++j) {
            for(int i=0;i<MR;++i) {
                AB[i + j*MR] = AB[i + j*MR] + A[i] * B[j];
            }
        }
        A = A + MR;
        B = B + NR;
    }

    // C = C + AB
    for(int j=0;j<NR;++j) {
        for(int i=0;i<MR;++i) {
            C[i*incColC + j*incRowC] = C[i*incColC + j*incRowC] + AB[i + j*MR];
            //printf("\t(%d %d) %5.3f\n",i,j,AB[i + j*MR]);
        }
    }
}

void dgemm_kernel_sse_asm(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    unsigned long long s = (KC-2) * 32;
    int msk1[2] = {0x01,0x0};
    int msk2[2] = {0x01,0x0};

    for(int l=0;l<MR*NR;++l) {
        AB[l] = 0.0;
    }

    __asm__ volatile
    (
    "                                      \n\t"
    "xor %%rcx, %%rcx                      \n\t"
    "                                      \n\t"
    "vxorpd %%xmm0, %%xmm0, %%xmm0         \n\t" // ab00,ab11
    "vxorpd %%xmm1, %%xmm1, %%xmm1         \n\t" // ab20,ab31
    "vxorpd %%xmm2, %%xmm2, %%xmm2         \n\t" // ab10,ab01
    "vxorpd %%xmm3, %%xmm3, %%xmm3         \n\t" // ab30,ab21
    "vxorpd %%xmm4, %%xmm4, %%xmm4         \n\t" // ab02,ab13
    "vxorpd %%xmm5, %%xmm5, %%xmm5         \n\t" // ab12,ab03
    "vxorpd %%xmm6, %%xmm6, %%xmm6         \n\t" // ab22,ab33
    "vxorpd %%xmm7, %%xmm7, %%xmm7         \n\t" // ab23,ab32
    "vxorpd %%xmm8, %%xmm8, %%xmm8         \n\t" //
    "vxorpd %%xmm9, %%xmm9, %%xmm9         \n\t" //
    "vxorpd %%xmm10, %%xmm10, %%xmm10      \n\t" //
    "vxorpd %%xmm11, %%xmm11, %%xmm11      \n\t" //
    "vxorpd %%xmm12, %%xmm12, %%xmm12      \n\t" //
    "vxorpd %%xmm13, %%xmm13, %%xmm13      \n\t" //
    "vxorpd %%xmm14, %%xmm14, %%xmm14      \n\t" //
    "vxorpd %%xmm15, %%xmm15, %%xmm15      \n\t" //
    "                                      \n\t"
    "vmovupd   (%[_a_], %%rcx), %%xmm8     \n\t" // load (a00,a10)
    "vmovupd 16(%[_a_], %%rcx), %%xmm9     \n\t" // load (a20,a30)
    "                                      \n\t"
    "vmovupd   (%[_b_], %%rcx), %%xmm10    \n\t" // load (b00,b01)
    "                                      \n\t"
    "1:                                    \n\t" // Loop Begin
    "                                      \n\t"
    "vmovupd 16(%[_b_], %%rcx), %%xmm11    \n\t" // load (b02,b03)
    "                                      \n\t"
    "vshufpd $0x01,%%xmm10,%%xmm10,%%xmm12 \n\t" // shuffle (b01,b00)
    "vshufpd $0x01,%%xmm11,%%xmm11,%%xmm13 \n\t" // shuffle (b03,b02)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm10,%%xmm0     \n\t" // FMADDPD (a00,a11)
    "vfmadd231pd %%xmm9,%%xmm10,%%xmm1     \n\t" // FMADDPD (a20,a31)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm12,%%xmm2     \n\t" // FMADDPD (a10,a01)
    "vfmadd231pd %%xmm9,%%xmm12,%%xmm3     \n\t" // FMADDPD (a30,a21)
    "                                      \n\t"
    "vmovupd 32(%[_b_], %%rcx), %%xmm10    \n\t" // load (b10,b11)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm4     \n\t" // FMADDPD (a02,a13)
    "vfmadd231pd %%xmm9,%%xmm11,%%xmm5     \n\t" // FMADDPD (a12,a03)
    "                                      \n\t"
    "vmovupd 32(%[_a_], %%rcx), %%xmm9     \n\t" // load (a01,a11)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm6     \n\t" // FMADDPD (a22,a33)
    "                                      \n\t"
    "vmovupd 48(%[_a_], %%rcx), %%xmm10    \n\t" // load (a21,a31)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm7     \n\t" // FMADDPD (a32,a23)
    "                                      \n\t"
    "add $32, %%rcx                        \n\t"
    "cmp %[_k_], %%rcx                     \n\t" 
    "jl  1b                                \n\t" // Loop End
    "                                      \n\t"
    "                                      \n\t"
    "movzx   (%[_m1_]), %%edx                  \n\t" // mask k1 (0, _ )
    "kmovb   %%edx, %%k1                      \n\t" // mask k1 (0, _ )
    //"movzx   (%[_m2_]), %%edx                  \n\t" // mask k2 ( _, 0)
    //"kmovb   %%edx, %%k2                      \n\t" // mask k2 ( _, 0)
    "vmovupd %%xmm0,  (%[_c_])               \n\t" // store (ab00)  
    "vmovupd %%xmm0,  (%[_c_])%{%%k1%}       \n\t" // store (ab00)  
    "vmovupd %%xmm2,8(%[_c_])%{%%k1%}        \n\t" // store (ab01)  2
    "vmovupd %%xmm4,16(%[_c_])%{%%k1%}       \n\t" // store (ab02)  
    "vmovupd %%xmm5,24(%[_c_])%{%%k1%}       \n\t" // store (ab03)  2
    "                                        \n\t"
    "vmovupd %%xmm2,32(%[_c_])%{%%k1%}       \n\t" // store (ab10)  
    "vmovupd %%xmm0,40(%[_c_])%{%%k1%}       \n\t" // store (ab11)  2
    "vmovupd %%xmm5,48(%[_c_])%{%%k1%}       \n\t" // store (ab12)  
    "vmovupd %%xmm4,56(%[_c_])%{%%k1%}       \n\t" // store (ab13)  2
    "                                        \n\t"
    "vmovupd %%xmm1,64(%[_c_])%{%%k1%}       \n\t" // store (ab20)  
    "vmovupd %%xmm3,72(%[_c_])%{%%k1%}       \n\t" // store (ab21)  2
    "vmovupd %%xmm6,80(%[_c_])%{%%k1%}       \n\t" // store (ab22)  
    "vmovupd %%xmm7,88(%[_c_])%{%%k1%}       \n\t" // store (ab23)  2
    "                                        \n\t"
    "vmovupd %%xmm3,96(%[_c_])%{%%k1%}       \n\t" // store (ab30)  
    "vmovupd %%xmm1,104(%[_c_])%{%%k1%}      \n\t" // store (ab31)  2
    "vmovupd %%xmm7,112(%[_c_])%{%%k1%}      \n\t" // store (ab32)  
    "vmovupd %%xmm6,120(%[_c_])%{%%k1%}      \n\t" // store (ab33)  2
    "                                      \n\t"
    "                                      \n\t"
    : // output
    : // input
      [_k_] "r"(s ),      // 0
      [_a_] "r"(A ),      // 1
      [_b_] "r"(B ),      // 2
      [_c_] "r"(AB ),     // 3
      [_m1_] "r"(msk1 ),     // 4
      [_m2_] "r"(msk2 )      // 5
    : // register clobber list
        "esi","rax", "rbx", "rcx", "k1", "k2",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15","memory"
    );

    // C = C + AB
    for(int j=0;j<NR;++j) {
        for(int i=0;i<MR;++i) {
            C[i*incColC + j*incRowC] = C[i*incColC + j*incRowC] + AB[i + j*MR];
            //printf("(%d %d) %5.3f\n",i,j,AB[i + j*MR]);
        }
    }
}

void dgemm_kernel_avx2_asm(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    unsigned long long s = (KC-2) * 32;
    int msk1[2] = {0x01,0x0};
    int msk2[2] = {0x01,0x0};

    for(int l=0;l<MR*NR;++l) {
        AB[l] = 0.0;
    }

    __asm__ volatile
    (
    "                                      \n\t"
    "xor %%rcx, %%rcx                      \n\t"
    "                                      \n\t"
    "vxorpd %%xmm0, %%xmm0, %%xmm0         \n\t" // ab00,ab11
    "vxorpd %%xmm1, %%xmm1, %%xmm1         \n\t" // ab20,ab31
    "vxorpd %%xmm2, %%xmm2, %%xmm2         \n\t" // ab10,ab01
    "vxorpd %%xmm3, %%xmm3, %%xmm3         \n\t" // ab30,ab21
    "vxorpd %%xmm4, %%xmm4, %%xmm4         \n\t" // ab02,ab13
    "vxorpd %%xmm5, %%xmm5, %%xmm5         \n\t" // ab12,ab03
    "vxorpd %%xmm6, %%xmm6, %%xmm6         \n\t" // ab22,ab33
    "vxorpd %%xmm7, %%xmm7, %%xmm7         \n\t" // ab23,ab32
    "vxorpd %%xmm8, %%xmm8, %%xmm8         \n\t" //
    "vxorpd %%xmm9, %%xmm9, %%xmm9         \n\t" //
    "vxorpd %%xmm10, %%xmm10, %%xmm10      \n\t" //
    "vxorpd %%xmm11, %%xmm11, %%xmm11      \n\t" //
    "vxorpd %%xmm12, %%xmm12, %%xmm12      \n\t" //
    "vxorpd %%xmm13, %%xmm13, %%xmm13      \n\t" //
    "vxorpd %%xmm14, %%xmm14, %%xmm14      \n\t" //
    "vxorpd %%xmm15, %%xmm15, %%xmm15      \n\t" //
    "                                      \n\t"
    "vmovupd   (%[_a_], %%rcx), %%xmm8     \n\t" // load (a00,a10)
    "vmovupd 16(%[_a_], %%rcx), %%xmm9     \n\t" // load (a20,a30)
    "                                      \n\t"
    "vmovupd   (%[_b_], %%rcx), %%xmm10    \n\t" // load (b00,b01)
    "                                      \n\t"
    "1:                                    \n\t" // Loop Begin
    "                                      \n\t"
    "vmovupd 16(%[_b_], %%rcx), %%xmm11    \n\t" // load (b02,b03)
    "                                      \n\t"
    "vshufpd $0x01,%%xmm10,%%xmm10,%%xmm12 \n\t" // shuffle (b01,b00)
    "vshufpd $0x01,%%xmm11,%%xmm11,%%xmm13 \n\t" // shuffle (b03,b02)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm10,%%xmm0     \n\t" // FMADDPD (a00,a11)
    "vfmadd231pd %%xmm9,%%xmm10,%%xmm1     \n\t" // FMADDPD (a20,a31)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm12,%%xmm2     \n\t" // FMADDPD (a10,a01)
    "vfmadd231pd %%xmm9,%%xmm12,%%xmm3     \n\t" // FMADDPD (a30,a21)
    "                                      \n\t"
    "vmovupd 32(%[_b_], %%rcx), %%xmm10    \n\t" // load (b10,b11)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm4     \n\t" // FMADDPD (a02,a13)
    "vfmadd231pd %%xmm9,%%xmm11,%%xmm5     \n\t" // FMADDPD (a12,a03)
    "                                      \n\t"
    "vmovupd 32(%[_a_], %%rcx), %%xmm9     \n\t" // load (a01,a11)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm6     \n\t" // FMADDPD (a22,a33)
    "                                      \n\t"
    "vmovupd 48(%[_a_], %%rcx), %%xmm10    \n\t" // load (a21,a31)
    "                                      \n\t"
    "vfmadd231pd %%xmm8,%%xmm11,%%xmm7     \n\t" // FMADDPD (a32,a23)
    "                                      \n\t"
    "add $32, %%rcx                        \n\t"
    "cmp %[_k_], %%rcx                     \n\t" 
    "jl  1b                                \n\t" // Loop End
    "                                      \n\t"
    "                                      \n\t"
    "movzx   (%[_m1_]), %%edx                  \n\t" // mask k1 (0, _ )
    "kmovb   %%edx, %%k1                      \n\t" // mask k1 (0, _ )
    //"movzx   (%[_m2_]), %%edx                  \n\t" // mask k2 ( _, 0)
    //"kmovb   %%edx, %%k2                      \n\t" // mask k2 ( _, 0)
    "vmovupd %%xmm0,  (%[_c_])               \n\t" // store (ab00)  
    "vmovupd %%xmm0,  (%[_c_])%{%%k1%}       \n\t" // store (ab00)  
    "vmovupd %%xmm2,8(%[_c_])%{%%k1%}        \n\t" // store (ab01)  2
    "vmovupd %%xmm4,16(%[_c_])%{%%k1%}       \n\t" // store (ab02)  
    "vmovupd %%xmm5,24(%[_c_])%{%%k1%}       \n\t" // store (ab03)  2
    "                                        \n\t"
    "vmovupd %%xmm2,32(%[_c_])%{%%k1%}       \n\t" // store (ab10)  
    "vmovupd %%xmm0,40(%[_c_])%{%%k1%}       \n\t" // store (ab11)  2
    "vmovupd %%xmm5,48(%[_c_])%{%%k1%}       \n\t" // store (ab12)  
    "vmovupd %%xmm4,56(%[_c_])%{%%k1%}       \n\t" // store (ab13)  2
    "                                        \n\t"
    "vmovupd %%xmm1,64(%[_c_])%{%%k1%}       \n\t" // store (ab20)  
    "vmovupd %%xmm3,72(%[_c_])%{%%k1%}       \n\t" // store (ab21)  2
    "vmovupd %%xmm6,80(%[_c_])%{%%k1%}       \n\t" // store (ab22)  
    "vmovupd %%xmm7,88(%[_c_])%{%%k1%}       \n\t" // store (ab23)  2
    "                                        \n\t"
    "vmovupd %%xmm3,96(%[_c_])%{%%k1%}       \n\t" // store (ab30)  
    "vmovupd %%xmm1,104(%[_c_])%{%%k1%}      \n\t" // store (ab31)  2
    "vmovupd %%xmm7,112(%[_c_])%{%%k1%}      \n\t" // store (ab32)  
    "vmovupd %%xmm6,120(%[_c_])%{%%k1%}      \n\t" // store (ab33)  2
    "                                      \n\t"
    "                                      \n\t"
    : // output
    : // input
      [_k_] "r"(s ),      // 0
      [_a_] "r"(A ),      // 1
      [_b_] "r"(B ),      // 2
      [_c_] "r"(AB ),     // 3
      [_m1_] "r"(msk1 ),     // 4
      [_m2_] "r"(msk2 )      // 5
    : // register clobber list
        "esi","rax", "rbx", "rcx", "k1", "k2",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15","memory"
    );

    // C = C + AB
    for(int j=0;j<NR;++j) {
        for(int i=0;i<MR;++i) {
            C[i*incColC + j*incRowC] = C[i*incColC + j*incRowC] + AB[i + j*MR];
            //printf("(%d %d) %5.3f\n",i,j,AB[i + j*MR]);
        }
    }
}

void dgemm_macro_kernel(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B) {
    int64_t mp = MC / MR;
    int64_t np = NC / NR;

    for(int j=0;j<np;++j) {
        for(int i=0;i<mp;++i) {
          //printf("--((%d %d))--\n",i,j);
            //dgemm_kernel(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[i*MR*incRowC + j*NR*incColC], incRowC, incColC);
            dgemm_kernel_sse_asm(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[i*MR*incRowC + j*NR*incColC], incRowC, incColC);
        }
    }
}

