#include "kernel.h"
#include "utils.h"

#include "bli_x86_asm_macros.h"

void dgemm_kernel_avx512_asm_unroll0(int64_t kc, double *  A, double *  B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc);


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

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 4), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 5), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 6), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 7), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 8), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 9), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(10), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(11), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(12), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(13), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(14), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(15), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(16), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(17), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(18), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(19), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(20), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(21), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(22), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(23), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(24), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(25), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(26), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(27), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(28), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(29), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
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
   [c] "m"(C)
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

void dgemm_kernel_avx512_asm_unroll2(int64_t kc, double *  A, double *  B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc >> 1);


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

      DEC(RSI)
      JNE(LOOP1)

    LABEL(K_LOOP)

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 4), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 5), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 6), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 7), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 8), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM( 9), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(10), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(11), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(12), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(13), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(14), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(15), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(16), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(17), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(18), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(19), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(20), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(21), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(22), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(23), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(24), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(25), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(26), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(27), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(28), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    VADDPD(ZMM( 1), ZMM(29), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
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
   [c] "m"(C)
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

void dgemm_kernel_avx512_asm_unroll4(int64_t kc, double *  A, double *  B, double *C, int64_t incRowC, int64_t incColC) {
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

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 4), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 5), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 6), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 7), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 8), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM( 9), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(10), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(11), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(12), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(13), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(14), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(15), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(16), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(17), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(18), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(19), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(20), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(21), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(22), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(23), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(24), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(25), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(26), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(27), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(28), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(29), MEM(RCX, 8*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 8*8), ZMM( 1)) // AB -> ZMM

    LEA(RCX, MEM(RCX,16*8))

    PREFETCH(0, MEM(RCX, 192*8)) // Preload B 0 - 1
    VADDPD(ZMM( 1), ZMM(30), MEM(RCX, 0*8)) // ZMM -> C
    VMOVUPD(MEM(RCX, 0*8), ZMM( 1)) // AB -> ZMM

    //PREFETCH(0, MEM(RCX,320*8)) // Preload B 0 - 1
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
   [c] "m"(C)
   : // clobber
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
  )

}

void dgemm_macro_kernel(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *  _A, double *  _B) {
    int64_t mp = mc / MR;
    int64_t np = nc / NR;
    int64_t KC = kc;
    int64_t nmcnc = 0;
    int64_t MRNR = MR*NR;
    int64_t MRKC = MR*KC;
    int64_t NRKC = NR*KC;
    int64_t KC64 = (KC/64) * 64;
    int64_t KC32 = ((kc - KC64)/32) * 32;
    int64_t KC16 = ((kc - KC64 - KC32)/16) * 16;
    int64_t MRKC64 = MR * KC64;
    int64_t NRKC64 = NR * KC64;
    int64_t MRKC32 = MR * KC32;
    int64_t NRKC32 = NR * KC32;
    int64_t MRKC16 = MR * KC16;
    int64_t NRKC16 = NR * KC16;
    double *_A_p;
    double *_B_p;
    double *_C_p;
    int i,j,k;

    switch (KC64){
      case 0:
        switch (KC32){
          case 0:
#pragma omp parallel 
{
#pragma omp for collapse(2) private(i)
            for(j=0;j<np;++j) {
//#pragma omp task 
//              {
                // Inc
                for(i=0;i<mp;++i) {
                    // Inc
                    //_C_p = C + nmcnc*MRNR;
                    dgemm_kernel_avx512_asm_unroll0(KC  , &_A[i*MRKC], &_B[j*NRKC], &C[(j*mp +i)*MRNR], incRowC, incColC);
                    //nmcnc = nmcnc + 1;
                }
//              }
            }
}
            break;
          default:
            if((kc-KC32) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
              for(j=0;j<np;++j) {
                  // Inc
                  for(i=0;i<mp;++i) {
                      // Inc
                      _C_p = C + nmcnc*MRNR;
                      dgemm_kernel_avx512_asm_unroll2(KC32, &_A[i*MRKC], &_B[j*NRKC], _C_p, incRowC, incColC);
                      nmcnc = nmcnc + 1;
                  }
              }
}
            }
            else if((kc-KC32-KC16) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
              for(j=0;j<np;++j) {
                  // Inc
                  for(i=0;i<mp;++i) {
                      // Inc
                      _C_p = C + nmcnc*MRNR;

                      dgemm_kernel_avx512_asm_unroll2(KC32, &_A[i*MRKC], &_B[j*NRKC], _C_p, incRowC, incColC);

                      dgemm_kernel_avx512_asm_unroll0(KC16, &_A[i*MRKC + KC32*MR], &_B[j*NRKC + KC32*NR], _C_p, incRowC, incColC);

                      nmcnc = nmcnc + 1;
                  }
              }
}
            }
            break;
        }
        break;
      default:
        if((kc-KC64) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
          for(j=0;j<np;++j) {
              for(i=0;i<mp;++i) {
                  dgemm_kernel_avx512_asm_unroll4(KC64, &_A[i*MRKC], &_B[j*NRKC], &C[(j*mp + i)*MRNR], incRowC, incColC);
              }
          }
}

        }
        else if((kc-KC64-KC32) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
          for(j=0;j<np;++j) {
              // Inc
              for(i=0;i<mp;++i) {
                  // Inc
                  _C_p = C + nmcnc*MRNR;

                  dgemm_kernel_avx512_asm_unroll4(KC64, &_A[i*MRKC], &_B[j*NRKC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx512_asm_unroll2(KC32, &_A[i*MRKC + MRKC64], &_B[j*NRKC + NRKC64], _C_p, incRowC, incColC);

                  nmcnc = nmcnc + 1;
              }
          }
}

        }
        else if((kc - KC64 - KC16) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
          for(j=0;j<np;++j) {
              // Inc
              for(i=0;i<mp;++i) {
                  // Inc
                  _C_p = C + nmcnc*MRNR;

                  dgemm_kernel_avx512_asm_unroll4(KC64, &_A[i*MRKC], &_B[j*NRKC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx512_asm_unroll0(KC16, &_A[i*MRKC + KC64*MR], &_B[j*NRKC + KC64*NR], _C_p, incRowC, incColC);

                  nmcnc = nmcnc + 1;
              }
          }
}

        }
        else if((kc - KC64 - KC32 - KC16) == 0){
#pragma omp parallel 
{
#pragma omp for private(i)
          for(j=0;j<np;++j) {
              // Inc
              for(i=0;i<mp;++i) {
                  // Inc
                  _C_p = C + nmcnc*MRNR;

                  dgemm_kernel_avx512_asm_unroll4(KC64, &_A[i*MRKC], &_B[j*NRKC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx512_asm_unroll2(KC32, &_A[i*MRKC + KC64*MR], &_B[j*NRKC + KC64*NR], _C_p, incRowC, incColC);

                  dgemm_kernel_avx512_asm_unroll0(KC16, &_A[i*MRKC + KC64*MR + KC32*MR], &_B[j*NRKC + KC64*NR + KC32*NR], _C_p, incRowC, incColC);

                  nmcnc = nmcnc + 1;
              }
          }
}

        }
        break;
    }
}

