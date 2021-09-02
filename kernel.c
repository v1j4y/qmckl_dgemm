#include "kernel.h"
#include "utils.h"

#include "bli_x86_asm_macros.h"

//void dgemm_kernel_avx512_asm(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
//    // AB = A * B
//    double AB[MR*NR] __attribute__ ((aligned(64)));
//    double *tmpA = A;
//    double *tmpB = B;
//    unsigned long long s = (KC-4) * 64;
//    int msk1[2] = {0x01,0x0};
//    int msk2[2] = {0x01,0x0};
//
//    for(int l=0;l<( MR * NR ) - 4;l = l + 4) {
//        AB[l + 0] = 0.0;
//        AB[l + 1] = 0.0;
//        AB[l + 2] = 0.0;
//        AB[l + 3] = 0.0;
//    }
//
//    __asm__ volatile
//    (
//    "                                      \n\t"
//    "xor %%rcx, %%rcx                      \n\t"
//    "                                      \n\t"
//    "vxorpd %%zmm0, %%zmm0, %%zmm0         \n\t" // ab00,ab11
//    "vxorpd %%zmm1, %%zmm1, %%zmm1         \n\t" // ab20,ab31
//    "vxorpd %%zmm2, %%zmm2, %%zmm2         \n\t" // ab10,ab01
//    "vxorpd %%zmm3, %%zmm3, %%zmm3         \n\t" // ab30,ab21
//    "vxorpd %%zmm4, %%zmm4, %%zmm4         \n\t" // ab02,ab13
//    "vxorpd %%zmm5, %%zmm5, %%zmm5         \n\t" // ab12,ab03
//    "vxorpd %%zmm6, %%zmm6, %%zmm6         \n\t" // ab22,ab33
//    "vxorpd %%zmm7, %%zmm7, %%zmm7         \n\t" // ab23,ab32
//    "vxorpd %%zmm8, %%zmm8, %%zmm8         \n\t" //
//    "vxorpd %%zmm9, %%zmm9, %%zmm9         \n\t" //
//    "vxorpd %%zmm10, %%zmm10, %%zmm10      \n\t" //
//    "vxorpd %%zmm11, %%zmm11, %%zmm11      \n\t" //
//    "vxorpd %%zmm12, %%zmm12, %%zmm12      \n\t" //
//    "vxorpd %%zmm13, %%zmm13, %%zmm13      \n\t" //
//    "vxorpd %%zmm14, %%zmm14, %%zmm14      \n\t" //
//    "vxorpd %%zmm15, %%zmm15, %%zmm15      \n\t" //
//    "                                      \n\t"
//    "vmovupd    (%[_a_], %%rcx), %%zmm1    \n\t" // load (b02,b03)
//    "                                      \n\t"
//    "vmovupd    (%[_b_], %%rcx), %%zmm2    \n\t" // load (b10,b11)
//    "                                      \n\t"
//    "1:                                    \n\t" // Loop Begin
//    "                                      \n\t"
//    //"vmovupd    (%[_a_], %%rcx), %%zmm1    \n\t" // load (b02,b03)
//    //"                                      \n\t"
//    //"vmovupd    (%[_b_], %%rcx), %%zmm2    \n\t" // load (b10,b11)
//    //"                                      \n\t"
//    //"                                      \n\t"
//    //"vmovupd  64(%[_a_], %%rcx), %%zmm3    \n\t" // load (b02,b03)
//    //"                                      \n\t"
//    //"vmovupd  64(%[_b_], %%rcx), %%zmm4    \n\t" // load (b10,b11)
//    //"                                      \n\t"
//    //"vmovupd 128(%[_a_], %%rcx), %%zmm5    \n\t" // load (b02,b03)
//    //"                                      \n\t"
//    //"vmovupd 128(%[_b_], %%rcx), %%zmm6    \n\t" // load (b10,b11)
//    //"                                      \n\t"
//    //"                                      \n\t"
//    //"vmovupd 196(%[_a_], %%rcx), %%zmm7    \n\t" // load (b02,b03)
//    //"                                      \n\t"
//    //"vmovupd 196(%[_b_], %%rcx), %%zmm8    \n\t" // load (b10,b11)
//    "                                      \n\t"
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm3     \n\t" // FMADDPD (a00,a11)
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm4     \n\t" // FMADDPD (a20,a31)
//    "                                      \n\t"
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm5     \n\t" // FMADDPD (a00,a11)
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm6     \n\t" // FMADDPD (a20,a31)
//    "                                      \n\t"
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm7     \n\t" // FMADDPD (a00,a11)
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm8     \n\t" // FMADDPD (a20,a31)
//    "                                      \n\t"
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm9     \n\t" // FMADDPD (a00,a11)
//    "vfmadd231pd %%xmm1,%%xmm2,%%xmm10    \n\t" // FMADDPD (a20,a31)
//    "                                      \n\t"
//    "add $256, %%rcx                        \n\t"
//    "cmp %[_k_], %%rcx                     \n\t" 
//    "jl  1b                                \n\t" // Loop End
//    "                                      \n\t"
//    "                                      \n\t"
//    //"movzx   (%[_m1_]), %%edx                  \n\t" // mask k1 (0, _ )
//    //"kmovb   %%edx, %%k1                      \n\t" // mask k1 (0, _ )
//    //"movzx   (%[_m2_]), %%edx                  \n\t" // mask k2 ( _, 0)
//    //"kmovb   %%edx, %%k2                      \n\t" // mask k2 ( _, 0)
//    //"vmovupd %%xmm0,  (%[_c_])               \n\t" // store (ab00)  
//    //"vmovupd %%xmm0,  (%[_c_])%{%%k1%}       \n\t" // store (ab00)  
//    //"vmovupd %%xmm2,8(%[_c_])%{%%k1%}        \n\t" // store (ab01)  2
//    //"vmovupd %%xmm4,16(%[_c_])%{%%k1%}       \n\t" // store (ab02)  
//    //"vmovupd %%xmm5,24(%[_c_])%{%%k1%}       \n\t" // store (ab03)  2
//    //"                                        \n\t"
//    //"vmovupd %%xmm2,32(%[_c_])%{%%k1%}       \n\t" // store (ab10)  
//    //"vmovupd %%xmm0,40(%[_c_])%{%%k1%}       \n\t" // store (ab11)  2
//    //"vmovupd %%xmm5,48(%[_c_])%{%%k1%}       \n\t" // store (ab12)  
//    //"vmovupd %%xmm4,56(%[_c_])%{%%k1%}       \n\t" // store (ab13)  2
//    //"                                        \n\t"
//    //"vmovupd %%xmm1,64(%[_c_])%{%%k1%}       \n\t" // store (ab20)  
//    //"vmovupd %%xmm3,72(%[_c_])%{%%k1%}       \n\t" // store (ab21)  2
//    //"vmovupd %%xmm6,80(%[_c_])%{%%k1%}       \n\t" // store (ab22)  
//    //"vmovupd %%xmm7,88(%[_c_])%{%%k1%}       \n\t" // store (ab23)  2
//    //"                                        \n\t"
//    //"vmovupd %%xmm3,96(%[_c_])%{%%k1%}       \n\t" // store (ab30)  
//    //"vmovupd %%xmm1,104(%[_c_])%{%%k1%}      \n\t" // store (ab31)  2
//    //"vmovupd %%xmm7,112(%[_c_])%{%%k1%}      \n\t" // store (ab32)  
//    //"vmovupd %%xmm6,120(%[_c_])%{%%k1%}      \n\t" // store (ab33)  2
//    "                                      \n\t"
//    "                                      \n\t"
//    : // output
//    : // input
//      [_k_] "r"(s ),      // 0
//      [_a_] "r"(A ),      // 1
//      [_b_] "r"(B ),      // 2
//      [_c_] "r"(AB ),     // 3
//      [_m1_] "r"(msk1 ),     // 4
//      [_m2_] "r"(msk2 )      // 5
//    : // register clobber list
//        "esi","rax", "rbx", "rcx", "k1", "k2",
//        "xmm0", "xmm1", "xmm2", "xmm3",
//        "xmm4", "xmm5", "xmm6", "xmm7",
//        "xmm8", "xmm9", "xmm10", "xmm11",
//        "xmm12", "xmm13", "xmm14", "xmm15","memory"
//    );
//
//
//    // C = C + AB
//    for(int j=0;j<NR;++j) {
//      double *cidxj = C + j*incRowC;
//      int    *idxlstj = idxlist + j * MR;
//        for(int i=0;i<(MR-8);i=i+8) {
//            int idx1 = idxlstj[i + 0];
//            int idx2 = idxlstj[i + 1];
//            int idx3 = idxlstj[i + 2];
//            int idx4 = idxlstj[i + 3];
//            int idx5 = idxlstj[i + 4];
//            int idx6 = idxlstj[i + 5];
//            int idx7 = idxlstj[i + 6];
//            int idx8 = idxlstj[i + 7];
//            cidxj[i + 0] = cidxj[i + 0] + AB[idx1];
//            cidxj[i + 1] = cidxj[i + 1] + AB[idx2];
//            cidxj[i + 2] = cidxj[i + 2] + AB[idx3];
//            cidxj[i + 3] = cidxj[i + 3] + AB[idx4];
//            cidxj[i + 4] = cidxj[i + 4] + AB[idx5];
//            cidxj[i + 5] = cidxj[i + 5] + AB[idx6];
//            cidxj[i + 6] = cidxj[i + 6] + AB[idx7];
//            cidxj[i + 7] = cidxj[i + 7] + AB[idx8];
//            //printf("(%d %d) %5.3f\n",i,j,AB[i + j*MR]);
//        }
//    }
//}

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

void proto_dgemm_asm(double *vecA, double *vecB, double *vecC, int64_t dimA, double *res) {

  uint64_t kl = (dimA >> 3) >> 1;
  double AB[16*14] __attribute__ ((aligned(64))) = {0.0};
  const uint64_t permid1[8] = {1,2,3,4,5,6,7,0};
  uint32_t msk=0xFFFF;
  //printf("kl=%ld\n",kl);

}

void dgemm_kernel_avx512_asm_store(double *C, double *AB, int64_t incRowC) {
    
    uint64_t kl = ( NR >> 0);

  BEGIN_ASM()

    VXORPD(ZMM( 0), ZMM( 0), ZMM( 0))
    VXORPD(ZMM( 1), ZMM( 1), ZMM( 1))

    MOV(RSI, VAR(k)) // Loop id
    MOV(RAX, VAR(a)) // Vec A
    MOV(RBX, VAR(b)) // Vec B
    MOV(RCX, VAR(s)) // incRowC

    TEST(RSI, RSI)
    JE(K_LOOP)

      LABEL(LOOP1)

        PREFETCH(1, MEM(RBX, RCX,8)) // Preload B 0 - 1
        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // AB -> ZMM
        VADDPD(ZMM( 1), ZMM( 0), MEM(RBX, 0*8)) // ZMM -> C
        VMOVUPD(MEM(RBX, 0*8), ZMM( 1)) // AB -> ZMM

        LEA(RAX, MEM(RAX,8*8))
        LEA(RBX, MEM(RBX,8*8))

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // AB -> ZMM
        VADDPD(ZMM( 1), ZMM( 0), MEM(RBX, 0*8)) // ZMM -> C
        VMOVUPD(MEM(RBX, 0*8), ZMM( 1)) // AB -> ZMM

        LEA(RAX, MEM(RAX,8*8))
        LEA(RBX, MEM(RBX,RCX,8))

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    VZEROUPPER()

  END_ASM 
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(AB),
   [b] "m"(C),
   [s] "m"(incRowC-8)
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
  )

    //for(int j=0;j<NR;++j) {
    //  //for(int i=0;i<MR;i = i+8) {
    //    //int i = 0;
    //    C[j*incRowC +  0] = C[j*incRowC +  0] + AB[j*MR +  0];
    //    C[j*incRowC +  1] = C[j*incRowC +  1] + AB[j*MR +  1];
    //    C[j*incRowC +  2] = C[j*incRowC +  2] + AB[j*MR +  2];
    //    C[j*incRowC +  3] = C[j*incRowC +  3] + AB[j*MR +  3];
    //    C[j*incRowC +  4] = C[j*incRowC +  4] + AB[j*MR +  4];
    //    C[j*incRowC +  5] = C[j*incRowC +  5] + AB[j*MR +  5];
    //    C[j*incRowC +  6] = C[j*incRowC +  6] + AB[j*MR +  6];
    //    C[j*incRowC +  7] = C[j*incRowC +  7] + AB[j*MR +  7];
    //    C[j*incRowC +  8] = C[j*incRowC +  8] + AB[j*MR +  8];
    //    C[j*incRowC +  9] = C[j*incRowC +  9] + AB[j*MR +  9];
    //    C[j*incRowC + 10] = C[j*incRowC + 10] + AB[j*MR + 10];
    //    C[j*incRowC + 11] = C[j*incRowC + 11] + AB[j*MR + 11];
    //    C[j*incRowC + 12] = C[j*incRowC + 12] + AB[j*MR + 12];
    //    C[j*incRowC + 13] = C[j*incRowC + 13] + AB[j*MR + 13];
    //    C[j*incRowC + 14] = C[j*incRowC + 14] + AB[j*MR + 14];
    //    C[j*incRowC + 15] = C[j*incRowC + 15] + AB[j*MR + 15];
    //  //}
    //}
    //for(int j=0;j<NR;++j) {
    //  for(int i=0;i<MR;++i) {
    //    printf(" %5.3f ",C[j*incRowC + i]);
    //  }
    //  printf("\n");
    //}
    //exit(0);
    // C = C + AB
    //for(int j=0;j<MR;++j) {
    //  double *cidxj = C + j*incRowC;
    //  cidxj[ 0] = cidxj[ 0] + AB[0*16 + j];
    //  cidxj[ 1] = cidxj[ 1] + AB[1*16 + j];
    //  cidxj[ 2] = cidxj[ 2] + AB[2*16 + j];
    //  cidxj[ 3] = cidxj[ 3] + AB[3*16 + j];
    //  cidxj[ 4] = cidxj[ 4] + AB[4*16 + j];
    //  cidxj[ 5] = cidxj[ 5] + AB[5*16 + j];
    //  cidxj[ 6] = cidxj[ 6] + AB[6*16 + j];
    //  cidxj[ 7] = cidxj[ 7] + AB[7*16 + j];
    //  cidxj[ 8] = cidxj[ 8] + AB[8*16 + j];
    //  cidxj[ 9] = cidxj[ 9] + AB[9*16 + j];
    //  cidxj[10] = cidxj[10] + AB[10*16 + j];
    //  cidxj[11] = cidxj[11] + AB[11*16 + j];
    //  cidxj[12] = cidxj[12] + AB[12*16 + j];
    //  cidxj[13] = cidxj[13] + AB[13*16 + j];
    //}
    //printf("\nMatrx A\n");
    //for(int i=0;i<kc;++i){
    //  for(int j=0;j<MR;++j){
    //    printf(" %5.3f ",A[i*MR + j]);
    //  }
    //  printf("\n");
    //}
    //printf("\nMatrx B\n");
    //for(int i=0;i<kc;++i){
    //  for(int j=0;j<NR;++j){
    //    printf(" %5.3f ",B[i*NR + j]);
    //  }
    //  printf("\n");
    //}
    //printf("Matrx AB\n");
    //print_matrix(AB,MR,NR);
    //exit(0);

}

void dgemm_kernel_avx512_asm_unroll0(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    double *tmpA = A;
    double *tmpB = B;
    uint64_t kl = (kc >> 0);

    //for(int l=0;l<( MR * NR ) - 4;l = l + 4) {
    //    AB[l + 0] = 0.0;
    //    AB[l + 1] = 0.0;
    //    AB[l + 2] = 0.0;
    //    AB[l + 3] = 0.0;
    //}


  BEGIN_ASM()

    VXORPD(ZMM( 0), ZMM( 0), ZMM( 0))
    VXORPD(ZMM( 1), ZMM( 1), ZMM( 1))
    VXORPD(ZMM( 2), ZMM( 2), ZMM( 2))
    VXORPD(ZMM( 3), ZMM( 3), ZMM( 3))
    VXORPD(ZMM( 4), ZMM( 4), ZMM( 4))
    VXORPD(ZMM( 5), ZMM( 5), ZMM( 5))
    VXORPD(ZMM( 6), ZMM( 6), ZMM( 6))
    VXORPD(ZMM( 7), ZMM( 7), ZMM( 7))
    VXORPD(ZMM( 8), ZMM( 8), ZMM( 8))
    VXORPD(ZMM( 9), ZMM( 9), ZMM( 9))
    VXORPD(ZMM(10), ZMM(10), ZMM(10))
    VXORPD(ZMM(11), ZMM(11), ZMM(11))
    VXORPD(ZMM(12), ZMM(12), ZMM(12))
    VXORPD(ZMM(13), ZMM(13), ZMM(13))
    VXORPD(ZMM(14), ZMM(14), ZMM(14))
    VXORPD(ZMM(15), ZMM(15), ZMM(15))
    VXORPD(ZMM(16), ZMM(16), ZMM(16))
    VXORPD(ZMM(17), ZMM(17), ZMM(17))
    VXORPD(ZMM(18), ZMM(18), ZMM(18))
    VXORPD(ZMM(19), ZMM(19), ZMM(19))
    VXORPD(ZMM(20), ZMM(20), ZMM(20))
    VXORPD(ZMM(21), ZMM(21), ZMM(21))
    VXORPD(ZMM(22), ZMM(22), ZMM(22))
    VXORPD(ZMM(23), ZMM(23), ZMM(23))
    VXORPD(ZMM(24), ZMM(24), ZMM(24))
    VXORPD(ZMM(25), ZMM(25), ZMM(25))
    VXORPD(ZMM(26), ZMM(26), ZMM(26))
    VXORPD(ZMM(27), ZMM(27), ZMM(27))
    VXORPD(ZMM(28), ZMM(28), ZMM(28))
    VXORPD(ZMM(29), ZMM(29), ZMM(29))
    VXORPD(ZMM(30), ZMM(30), ZMM(30))
    VXORPD(ZMM(31), ZMM(31), ZMM(31))

    MOV(RSI, VAR(k)) // Loop id
    MOV(RAX, VAR(a)) // Vec A
    MOV(RBX, VAR(b)) // Vec B
    MOV(RCX, VAR(c)) // Vec B

    TEST(RSI, RSI)
    JE(K_LOOP)

      LABEL(LOOP1)

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    VMOVUPD(MEM(RCX, 0*8), ZMM( 4)) // Store res in C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 5)) // Store res in C
    VMOVUPD(MEM(RCX,16*8), ZMM( 6)) // Store res in C
    VMOVUPD(MEM(RCX,24*8), ZMM( 7)) // Store res in C
    VMOVUPD(MEM(RCX,32*8), ZMM( 8)) // Store res in C
    VMOVUPD(MEM(RCX,40*8), ZMM( 9)) // Store res in C
    VMOVUPD(MEM(RCX,48*8), ZMM(10)) // Store res in C
    VMOVUPD(MEM(RCX,56*8), ZMM(11)) // Store res in C
    VMOVUPD(MEM(RCX,64*8), ZMM(12)) // Store res in C
    VMOVUPD(MEM(RCX,72*8), ZMM(13)) // Store res in C
    VMOVUPD(MEM(RCX,80*8), ZMM(14)) // Store res in C
    VMOVUPD(MEM(RCX,88*8), ZMM(15)) // Store res in C
    VMOVUPD(MEM(RCX,96*8), ZMM(16)) // Store res in C
    VMOVUPD(MEM(RCX,104*8), ZMM(17)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 +  0*8), ZMM(18)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 +  8*8), ZMM(19)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 16*8), ZMM(20)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 24*8), ZMM(21)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 32*8), ZMM(22)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 40*8), ZMM(23)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 48*8), ZMM(24)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 56*8), ZMM(25)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 64*8), ZMM(26)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 72*8), ZMM(27)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 80*8), ZMM(28)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 88*8), ZMM(29)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 96*8), ZMM(30)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 104*8), ZMM(31)) // Store res in C

    VZEROUPPER()

  END_ASM 
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(A),
   [b] "m"(B),
   [c] "m"(AB)
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
  )

    dgemm_kernel_avx512_asm_store(C, AB, incRowC);

}

void dgemm_kernel_avx512_asm_unroll2(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    double *tmpA = A;
    double *tmpB = B;
    uint64_t kl = (kc >> 1);

    //for(int l=0;l<( MR * NR ) - 4;l = l + 4) {
    //    AB[l + 0] = 0.0;
    //    AB[l + 1] = 0.0;
    //    AB[l + 2] = 0.0;
    //    AB[l + 3] = 0.0;
    //}


  BEGIN_ASM()

    VXORPD(ZMM( 0), ZMM( 0), ZMM( 0))
    VXORPD(ZMM( 1), ZMM( 1), ZMM( 1))
    VXORPD(ZMM( 2), ZMM( 2), ZMM( 2))
    VXORPD(ZMM( 3), ZMM( 3), ZMM( 3))
    VXORPD(ZMM( 4), ZMM( 4), ZMM( 4))
    VXORPD(ZMM( 5), ZMM( 5), ZMM( 5))
    VXORPD(ZMM( 6), ZMM( 6), ZMM( 6))
    VXORPD(ZMM( 7), ZMM( 7), ZMM( 7))
    VXORPD(ZMM( 8), ZMM( 8), ZMM( 8))
    VXORPD(ZMM( 9), ZMM( 9), ZMM( 9))
    VXORPD(ZMM(10), ZMM(10), ZMM(10))
    VXORPD(ZMM(11), ZMM(11), ZMM(11))
    VXORPD(ZMM(12), ZMM(12), ZMM(12))
    VXORPD(ZMM(13), ZMM(13), ZMM(13))
    VXORPD(ZMM(14), ZMM(14), ZMM(14))
    VXORPD(ZMM(15), ZMM(15), ZMM(15))
    VXORPD(ZMM(16), ZMM(16), ZMM(16))
    VXORPD(ZMM(17), ZMM(17), ZMM(17))
    VXORPD(ZMM(18), ZMM(18), ZMM(18))
    VXORPD(ZMM(19), ZMM(19), ZMM(19))
    VXORPD(ZMM(20), ZMM(20), ZMM(20))
    VXORPD(ZMM(21), ZMM(21), ZMM(21))
    VXORPD(ZMM(22), ZMM(22), ZMM(22))
    VXORPD(ZMM(23), ZMM(23), ZMM(23))
    VXORPD(ZMM(24), ZMM(24), ZMM(24))
    VXORPD(ZMM(25), ZMM(25), ZMM(25))
    VXORPD(ZMM(26), ZMM(26), ZMM(26))
    VXORPD(ZMM(27), ZMM(27), ZMM(27))
    VXORPD(ZMM(28), ZMM(28), ZMM(28))
    VXORPD(ZMM(29), ZMM(29), ZMM(29))
    VXORPD(ZMM(30), ZMM(30), ZMM(30))
    VXORPD(ZMM(31), ZMM(31), ZMM(31))

    MOV(RSI, VAR(k)) // Loop id
    MOV(RAX, VAR(a)) // Vec A
    MOV(RBX, VAR(b)) // Vec B
    MOV(RCX, VAR(c)) // Vec B

    TEST(RSI, RSI)
    JE(K_LOOP)

      LABEL(LOOP1)

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RAX, 16*8)) // Preload B 0 - 1

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 14*8)) // Preload B 0 - 1

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RAX, 16*8)) // Preload B 0 - 1

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 14*8)) // Preload B 0 - 1

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    VMOVUPD(MEM(RCX, 0*8), ZMM( 4)) // Store res in C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 5)) // Store res in C
    VMOVUPD(MEM(RCX,16*8), ZMM( 6)) // Store res in C
    VMOVUPD(MEM(RCX,24*8), ZMM( 7)) // Store res in C
    VMOVUPD(MEM(RCX,32*8), ZMM( 8)) // Store res in C
    VMOVUPD(MEM(RCX,40*8), ZMM( 9)) // Store res in C
    VMOVUPD(MEM(RCX,48*8), ZMM(10)) // Store res in C
    VMOVUPD(MEM(RCX,56*8), ZMM(11)) // Store res in C
    VMOVUPD(MEM(RCX,64*8), ZMM(12)) // Store res in C
    VMOVUPD(MEM(RCX,72*8), ZMM(13)) // Store res in C
    VMOVUPD(MEM(RCX,80*8), ZMM(14)) // Store res in C
    VMOVUPD(MEM(RCX,88*8), ZMM(15)) // Store res in C
    VMOVUPD(MEM(RCX,96*8), ZMM(16)) // Store res in C
    VMOVUPD(MEM(RCX,104*8), ZMM(17)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 +  0*8), ZMM(18)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 +  8*8), ZMM(19)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 16*8), ZMM(20)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 24*8), ZMM(21)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 32*8), ZMM(22)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 40*8), ZMM(23)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 48*8), ZMM(24)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 56*8), ZMM(25)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 64*8), ZMM(26)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 72*8), ZMM(27)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 80*8), ZMM(28)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 88*8), ZMM(29)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 96*8), ZMM(30)) // Store res in C
    VMOVUPD(MEM(RCX,112*8 + 104*8), ZMM(31)) // Store res in C

    VZEROUPPER()

  END_ASM 
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(A),
   [b] "m"(B),
   [c] "m"(AB)
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
  )

    dgemm_kernel_avx512_asm_store(C, AB, incRowC);

}

void dgemm_kernel_avx512_asm_unroll4(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    //double AB[MR*MR] __attribute__ ((aligned(64)));
    double *tmpA = A;
    double *tmpB = B;
    uint64_t kl = (kc >> 2);


  BEGIN_ASM()

    VXORPD(ZMM( 4), ZMM( 4), ZMM( 4))
    VXORPD(ZMM( 5), ZMM( 5), ZMM( 5))
    VXORPD(ZMM( 6), ZMM( 6), ZMM( 6))
    VXORPD(ZMM( 7), ZMM( 7), ZMM( 7))
    VXORPD(ZMM( 8), ZMM( 8), ZMM( 8))
    VXORPD(ZMM( 9), ZMM( 9), ZMM( 9))
    VXORPD(ZMM(10), ZMM(10), ZMM(10))
    VXORPD(ZMM(11), ZMM(11), ZMM(11))
    VXORPD(ZMM(12), ZMM(12), ZMM(12))
    VXORPD(ZMM(13), ZMM(13), ZMM(13))
    VXORPD(ZMM(14), ZMM(14), ZMM(14))
    VXORPD(ZMM(15), ZMM(15), ZMM(15))
    VXORPD(ZMM(16), ZMM(16), ZMM(16))
    VXORPD(ZMM(17), ZMM(17), ZMM(17))
    VXORPD(ZMM(18), ZMM(18), ZMM(18))
    VXORPD(ZMM(19), ZMM(19), ZMM(19))
    VXORPD(ZMM(20), ZMM(20), ZMM(20))
    VXORPD(ZMM(21), ZMM(21), ZMM(21))
    VXORPD(ZMM(22), ZMM(22), ZMM(22))
    VXORPD(ZMM(23), ZMM(23), ZMM(23))
    VXORPD(ZMM(24), ZMM(24), ZMM(24))
    VXORPD(ZMM(25), ZMM(25), ZMM(25))
    VXORPD(ZMM(26), ZMM(26), ZMM(26))
    VXORPD(ZMM(27), ZMM(27), ZMM(27))
    VXORPD(ZMM(28), ZMM(28), ZMM(28))
    VXORPD(ZMM(29), ZMM(29), ZMM(29))
    VXORPD(ZMM(30), ZMM(30), ZMM(30))
    VXORPD(ZMM(31), ZMM(31), ZMM(31))

    MOV(RSI, VAR(k)) // Loop id
    MOV(RAX, VAR(a)) // Vec A
    MOV(RBX, VAR(b)) // Vec B
    MOV(RCX, VAR(c)) // Vec C
    MOV(RDX, VAR(d)) // Step C
    MOV( R8, VAR(e)) // Step C

    TEST(RSI, RSI)
    JE(K_LOOP)

      LABEL(LOOP1)

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        //PREFETCH(0, MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 16*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 20*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 20*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

        VMOVUPD(ZMM( 0), MEM(RAX, 0*8)) // Preload A 0 - 8
        VMOVUPD(ZMM( 1), MEM(RAX, 8*8)) // Preload A 9 - 15

        VBROADCASTSD(ZMM( 2), MEM(RBX, 0*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 1*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 4), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 5), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 6), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 7), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 2*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 3*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM( 8), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM( 9), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(10), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(11), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 4*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 5*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(12), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(13), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(14), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(15), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        //PREFETCH(0, MEM(RBX, 14*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 2), MEM(RBX, 6*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 7*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(16), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(17), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(18), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(19), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX, 8*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX, 9*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(20), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(21), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(22), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(23), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,10*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,11*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(24), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(25), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(26), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(27), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        VBROADCASTSD(ZMM( 2), MEM(RBX,12*8)) // Preload B 0 - 1
        VBROADCASTSD(ZMM( 3), MEM(RBX,13*8)) // Preload B 0 - 1
        VFMADD231PD(ZMM(28), ZMM( 0), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(29), ZMM( 1), ZMM( 2)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(30), ZMM( 0), ZMM( 3)) // FMA and save to C zmm4 - zmm7
        VFMADD231PD(ZMM(31), ZMM( 1), ZMM( 3)) // FMA and save to C zmm4 - zmm7

        LEA(RAX, MEM(RAX,16*8))
        LEA(RBX, MEM(RBX,14*8))

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 4), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 5), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM( 6), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 7), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 8), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 9), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(10), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(11), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(12), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(13), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(14), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(15), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(16), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(17), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(18), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(19), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(20), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(21), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(22), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(23), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(24), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(25), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(26), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(27), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    //PREFETCH(1, MEM(RCX,  R8,8)) // Preload B 0 - 1
    PREFETCH(0, MEM(RCX, 32*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(28), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(29), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    VADDPD(ZMM( 1), ZMM(30), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(31), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM


    VZEROUPPER()

  END_ASM 
  (
   : // output
   : // input
   [k] "m"(kl),
   [a] "m"(A),
   [b] "m"(B),
   [c] "m"(C),
   [d] "m"(incRowC),
   [e] "m"((incRowC)*2)
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
  )

    //dgemm_kernel_avx512_asm_store(C, AB, incRowC);

}

void dgemm_kernel_avx512_mipp(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    // AB = A * B
    double AB[MR*NR] __attribute__ ((aligned(64)));
    double *tmpA = A;
    double *tmpB = B;

    for(int l=0;l<( MR * NR ) - 4;l = l + 4) {
        AB[l + 0] = 0.0;
        AB[l + 1] = 0.0;
        AB[l + 2] = 0.0;
        AB[l + 3] = 0.0;
    }

    // Cols of AB in SSE registers
    __m512d   ab_00;
    __m512d   ab_01;
    __m512d   ab_02;
    __m512d   ab_03;
    __m512d   ab_04;
    __m512d   ab_05;
    //mipp::Reg<double>   ab_00;
    //mipp::Reg<double>   ab_01;
    //mipp::Reg<double>   ab_02;
    //mipp::Reg<double>   ab_03;
    //mipp::Reg<double>   ab_04;
    //mipp::Reg<double>   ab_05;
    //mipp::Reg<double>   ab_06;
    //mipp::Reg<double>   ab_07;
    //mipp::Reg<double>   ab_08;
    //mipp::Reg<double>   ab_09;
    //mipp::Reg<double>   ab_10;
    //mipp::Reg<double>   ab_11;
    //mipp::Reg<double>   ab_12;
    //mipp::Reg<double>   ab_13;
    //mipp::Reg<double>   ab_14;
    //mipp::Reg<double>   ab_15;
    //mipp::Reg<double>   ab_16;
    //mipp::Reg<double>   ab_17;
    //mipp::Reg<double>   ab_18;
    //mipp::Reg<double>   ab_19;
    //mipp::Reg<double>   ab_20;
    //mipp::Reg<double>   ab_21;
    //mipp::Reg<double>   ab_22;
    //mipp::Reg<double>   ab_23;
    //mipp::Reg<double>   ab_24;
    //mipp::Reg<double>   ab_25;
    //mipp::Reg<double>   ab_26;
    //mipp::Reg<double>   ab_27;

    __m512d   a_00;
    __m512d   a_01;
    __m512d   tmp1, tmp2;
    __m512d   tmp3;//, tmp4;
    //mipp::Reg<double>   a_00;
    //mipp::Reg<double>   a_01;
    //mipp::Reg<double>   tmp1, tmp2;
    //mipp::Reg<double>   tmp3;//, tmp4;
    //mipp::Reg<double>   tmp5, tmp6;
    //mipp::Reg<double>   tmp7, tmp8;
    //mipp::Reg<double>   tmp9, tmp10;
    //mipp::Reg<double>   tmp11, tmp12;
    //mipp::Reg<double>   tmp13, tmp14;
    //mipp::Reg<double>   tmp15, tmp16;

    int rD = mipp::N<double>();

    // Set them to 0
    ab_00 = _mm512_setzero_pd();
    ab_01 = _mm512_setzero_pd();
    ab_02 = _mm512_setzero_pd();
    ab_03 = _mm512_setzero_pd();
    ab_04 = _mm512_setzero_pd();
    ab_05 = _mm512_setzero_pd();
    //ab_00 = 0.0;
    //ab_01 = 0.0;
    //ab_02 = 0.0;
    //ab_03 = 0.0;
    //ab_04 = 0.0;
    //ab_05 = 0.0;
    //ab_06 = 0.0;
    //ab_07 = 0.0;
    //ab_08 = 0.0;
    //ab_09 = 0.0;
    //ab_10 = 0.0;
    //ab_11 = 0.0;
    //ab_12 = 0.0;
    //ab_13 = 0.0;
    //ab_14 = 0.0;
    //ab_15 = 0.0;
    //ab_16 = 0.0;
    //ab_17 = 0.0;
    //ab_18 = 0.0;
    //ab_19 = 0.0;
    //ab_20 = 0.0;
    //ab_21 = 0.0;
    //ab_22 = 0.0;
    //ab_23 = 0.0;
    //ab_24 = 0.0;
    //ab_25 = 0.0;
    //ab_26 = 0.0;
    //ab_27 = 0.0;

    for(int k=0;k<(kc);++k) {
        a_00 = _mm512_load_pd(A + 0); // Preload A 0 - 8
        a_01 = _mm512_load_pd(A + 8); // Preload A 9 - 15
        //a_00.load(A + 0); // Preload A 0 - 8
        //a_01.load(A + 8); // Preload A 9 - 15

        //tmp1 = B[0]; // Preload B 0 - 1
        //tmp2 = B[1]; // Preload B 0 - 1
        tmp1 = _mm512_set1_pd(B[0]); // Preload B 0 - 1
        tmp2 = _mm512_set1_pd(B[1]); // Preload B 0 - 1
        //ab_00 = mipp::fmadd(a_00, tmp1, ab_00); // FMA and save to C zmm4 - zmm7
        //ab_01 = mipp::fmadd(a_01, tmp1, ab_01); // FMA and save to C zmm4 - zmm7
        //ab_02 = mipp::fmadd(a_00, tmp2, ab_02); // FMA and save to C zmm4 - zmm7
        //ab_03 = mipp::fmadd(a_01, tmp2, ab_03); // FMA and save to C zmm4 - zmm7
        ab_00 = _mm512_fmadd_pd(a_00, tmp1, ab_00); // FMA and save to C zmm4 - zmm7
        ab_01 = _mm512_fmadd_pd(a_01, tmp1, ab_01); // FMA and save to C zmm4 - zmm7
        ab_02 = _mm512_fmadd_pd(a_00, tmp2, ab_02); // FMA and save to C zmm4 - zmm7
        ab_03 = _mm512_fmadd_pd(a_01, tmp2, ab_03); // FMA and save to C zmm4 - zmm7

        //tmp3 = B[2]; // Preload B 0 - 1
        tmp3 = _mm512_set1_pd(B[2]); // Preload B 0 - 1
        //tmp4 = B[3]; // Preload B 0 - 1
        //ab_04 = mipp::fmadd(a_00, tmp3, ab_04); // FMA and save to C zmm4 - zmm7
        //ab_05 = mipp::fmadd(a_01, tmp3, ab_05); // FMA and save to C zmm4 - zmm7
        ab_04 = _mm512_fmadd_pd(a_00, tmp3, ab_04); // FMA and save to C zmm4 - zmm7
        ab_05 = _mm512_fmadd_pd(a_01, tmp3, ab_05); // FMA and save to C zmm4 - zmm7
        //ab_06 = mipp::fmadd(a_00, tmp4, ab_06); // FMA and save to C zmm4 - zmm7
        //ab_07 = mipp::fmadd(a_01, tmp4, ab_07); // FMA and save to C zmm4 - zmm7

        //tmp1 = B[4]; // Preload B 0 - 1
        //tmp2 = B[5]; // Preload B 0 - 1
        //ab_08 = mipp::fmadd(a_00, tmp1, ab_08); // FMA and save to C zmm4 - zmm7
        //ab_09 = mipp::fmadd(a_01, tmp1, ab_09); // FMA and save to C zmm4 - zmm7
        //ab_10 = mipp::fmadd(a_00, tmp2, ab_10); // FMA and save to C zmm4 - zmm7
        //ab_11 = mipp::fmadd(a_01, tmp2, ab_11); // FMA and save to C zmm4 - zmm7

        //tmp1 = B[6]; // Preload B 0 - 1
        //tmp2 = B[7]; // Preload B 0 - 1
        //ab_12 = mipp::fmadd(a_00, tmp1, ab_12); // FMA and save to C zmm4 - zmm7
        //ab_13 = mipp::fmadd(a_01, tmp1, ab_13); // FMA and save to C zmm4 - zmm7
        //ab_14 = mipp::fmadd(a_00, tmp2, ab_14); // FMA and save to C zmm4 - zmm7
        //ab_15 = mipp::fmadd(a_01, tmp2, ab_15); // FMA and save to C zmm4 - zmm7

        //tmp1 = B[8]; // Preload B 0 - 1
        //tmp2 = B[9]; // Preload B 0 - 1
        //ab_16 = mipp::fmadd(a_00, tmp1, ab_16); // FMA and save to C zmm4 - zmm7
        //ab_17 = mipp::fmadd(a_01, tmp1, ab_17); // FMA and save to C zmm4 - zmm7
        //ab_18 = mipp::fmadd(a_00, tmp2, ab_18); // FMA and save to C zmm4 - zmm7
        //ab_19 = mipp::fmadd(a_01, tmp2, ab_19); // FMA and save to C zmm4 - zmm7

        //tmp1 = B[10]; // Preload B 0 - 1
        //tmp2 = B[11]; // Preload B 0 - 1
        //ab_20 = mipp::fmadd(a_00, tmp1, ab_20); // FMA and save to C zmm4 - zmm7
        //ab_21 = mipp::fmadd(a_01, tmp1, ab_21); // FMA and save to C zmm4 - zmm7
        //ab_22 = mipp::fmadd(a_00, tmp2, ab_22); // FMA and save to C zmm4 - zmm7
        //ab_23 = mipp::fmadd(a_01, tmp2, ab_23); // FMA and save to C zmm4 - zmm7

        //tmp1 = B[12]; // Preload B 0 - 1
        //tmp2 = B[13]; // Preload B 0 - 1
        //ab_24 = mipp::fmadd(a_00, tmp1, ab_24); // FMA and save to C zmm4 - zmm7
        //ab_25 = mipp::fmadd(a_01, tmp1, ab_25); // FMA and save to C zmm4 - zmm7
        //ab_26 = mipp::fmadd(a_00, tmp2, ab_26); // FMA and save to C zmm4 - zmm7
        //ab_27 = mipp::fmadd(a_01, tmp2, ab_27); // FMA and save to C zmm4 - zmm7

      A = A + 16;
      B = B + 14;
    }


    _mm512_store_pd(AB +  0, ab_00); // Store res in C
    _mm512_store_pd(AB +  8, ab_01); // Store res in C
    _mm512_store_pd(AB + 16, ab_02); // Store res in C
    _mm512_store_pd(AB + 24, ab_03); // Store res in C
    _mm512_store_pd(AB + 32, ab_04); // Store res in C
    _mm512_store_pd(AB + 40, ab_05); // Store res in C
    //ab_00.store(AB +  0); // Store res in C
    //ab_01.store(AB +  8); // Store res in C
    //ab_02.store(AB + 16); // Store res in C
    //ab_03.store(AB + 24); // Store res in C
    //ab_04.store(AB + 32); // Store res in C
    //ab_05.store(AB + 40); // Store res in C
    //ab_06.store(AB + 48); // Store res in C
    //ab_07.store(AB + 56); // Store res in C
    //ab_08.store(AB + 64); // Store res in C
    //ab_09.store(AB + 72); // Store res in C
    //ab_10.store(AB + 80); // Store res in C
    //ab_11.store(AB + 88); // Store res in C
    //ab_12.store(AB + 96); // Store res in C
    //ab_13.store(AB + 104); // Store res in C
    //ab_14.store(AB + 112 +  0); // Store res in C
    //ab_15.store(AB + 112 +  8); // Store res in C
    //ab_16.store(AB + 112 + 16); // Store res in C
    //ab_17.store(AB + 112 + 24); // Store res in C
    //ab_18.store(AB + 112 + 32); // Store res in C
    //ab_19.store(AB + 112 + 40); // Store res in C
    //ab_20.store(AB + 112 + 48); // Store res in C
    //ab_21.store(AB + 112 + 56); // Store res in C
    //ab_22.store(AB + 112 + 64); // Store res in C
    //ab_23.store(AB + 112 + 72); // Store res in C
    //ab_24.store(AB + 112 + 80); // Store res in C
    //ab_25.store(AB + 112 + 88); // Store res in C
    //ab_26.store(AB + 112 + 96); // Store res in C
    //ab_27.store(AB + 112 + 104); // Store res in C


    // C = C + AB
    for(int j=0;j<MR;++j) {
      double *cidxj = C + j*incRowC;
      //int    *idxlstj = idxlist + j * MR;
        //for(int i=0;i<(MR-8);i=i+8) {
            //int idx1 = idxlstj[i + 0];
            //int idx2 = idxlstj[i + 1];
            //int idx3 = idxlstj[i + 2];
            //int idx4 = idxlstj[i + 3];
            //int idx5 = idxlstj[i + 4];
            //int idx6 = idxlstj[i + 5];
            //int idx7 = idxlstj[i + 6];
            //int idx8 = idxlstj[i + 7];
            cidxj[ 0] = cidxj[ 0] + AB[0*16 + j];
            cidxj[ 1] = cidxj[ 1] + AB[1*16 + j];
            cidxj[ 2] = cidxj[ 2] + AB[2*16 + j];
            cidxj[ 3] = cidxj[ 3] + AB[3*16 + j];
            cidxj[ 4] = cidxj[ 4] + AB[4*16 + j];
            cidxj[ 5] = cidxj[ 5] + AB[5*16 + j];
            cidxj[ 6] = cidxj[ 6] + AB[6*16 + j];
            cidxj[ 7] = cidxj[ 7] + AB[7*16 + j];
            cidxj[ 8] = cidxj[ 8] + AB[8*16 + j];
            cidxj[ 9] = cidxj[ 9] + AB[9*16 + j];
            cidxj[10] = cidxj[10] + AB[10*16 + j];
            cidxj[11] = cidxj[11] + AB[11*16 + j];
            cidxj[12] = cidxj[12] + AB[12*16 + j];
            cidxj[13] = cidxj[13] + AB[13*16 + j];
            //printf("(%d %d) %5.3f\n",i,j,AB[i + j*MR]);
        //}
    }
}

void dgemm_macro_kernel(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *_A, double *_B) {
    int64_t mp = MC / MR;
    int64_t np = NC / NR;
    int64_t nmcnc = 0;
    int64_t MRNR = MR*NR;

    for(int j=0;j<np;++j) {
        for(int i=0;i<mp;++i) {
            //dgemm_kernel(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[i*MR*incRowC + j*NR*incColC], incRowC, incColC);
            //dgemm_kernel_sse_asm(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[i*MR*incRowC + j*NR*incColC], incRowC, incColC);
            //dgemm_kernel_avx512_mipp(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[i*MR*incRowC + j*NR*incColC], incRowC, incColC);
            //dgemm_kernel_avx512_asm_unroll0(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[j*NR*incRowC + i*MR*incColC], incRowC, incColC);
            dgemm_kernel_avx512_asm_unroll4(kc, &_A[i*MR*kc], &_B[j*NR*kc], &C[nmcnc*MRNR], incRowC, incColC);
            nmcnc = nmcnc + 1;
          //printf("(%d %d) %5.3f\n",i,j, C[0]);
        }
    }
}

