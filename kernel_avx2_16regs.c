#include "kernel_avx2_16regs.h"
#include "utils.h"

#include "bli_x86_asm_macros.h"


void dgemm_kernel_avx2_16regs_asm_unroll0(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc);

  BEGIN_ASM()
VXORPD(YMM( 0), YMM( 0), YMM( 0))
VXORPD(YMM( 1), YMM( 1), YMM( 1))
VXORPD(YMM( 2), YMM( 2), YMM( 2))
VXORPD(YMM( 3), YMM( 3), YMM( 3))
VXORPD(YMM( 4), YMM( 4), YMM( 4))
VXORPD(YMM( 5), YMM( 5), YMM( 5))
VXORPD(YMM( 6), YMM( 6), YMM( 6))
VXORPD(YMM( 7), YMM( 7), YMM( 7))
VXORPD(YMM( 8), YMM( 8), YMM( 8))
VXORPD(YMM( 9), YMM( 9), YMM( 9))
VXORPD(YMM( 10), YMM( 10), YMM( 10))
VXORPD(YMM( 11), YMM( 11), YMM( 11))
VXORPD(YMM( 12), YMM( 12), YMM( 12))
VXORPD(YMM( 13), YMM( 13), YMM( 13))
VXORPD(YMM( 14), YMM( 14), YMM( 14))
VXORPD(YMM( 15), YMM( 15), YMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 5), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 7), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 9), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 11), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 13), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 15), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))


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
          "r13", "r14", "r15", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",  "memory"
  )



}

void dgemm_kernel_avx2_16regs_asm_unroll2(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc >> 1);

  BEGIN_ASM()
VXORPD(YMM( 0), YMM( 0), YMM( 0))
VXORPD(YMM( 1), YMM( 1), YMM( 1))
VXORPD(YMM( 2), YMM( 2), YMM( 2))
VXORPD(YMM( 3), YMM( 3), YMM( 3))
VXORPD(YMM( 4), YMM( 4), YMM( 4))
VXORPD(YMM( 5), YMM( 5), YMM( 5))
VXORPD(YMM( 6), YMM( 6), YMM( 6))
VXORPD(YMM( 7), YMM( 7), YMM( 7))
VXORPD(YMM( 8), YMM( 8), YMM( 8))
VXORPD(YMM( 9), YMM( 9), YMM( 9))
VXORPD(YMM( 10), YMM( 10), YMM( 10))
VXORPD(YMM( 11), YMM( 11), YMM( 11))
VXORPD(YMM( 12), YMM( 12), YMM( 12))
VXORPD(YMM( 13), YMM( 13), YMM( 13))
VXORPD(YMM( 14), YMM( 14), YMM( 14))
VXORPD(YMM( 15), YMM( 15), YMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 5), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 7), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 9), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 11), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 13), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 15), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))


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
          "r13", "r14", "r15", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",  "memory"
  )



}

void dgemm_kernel_avx2_16regs_asm_unroll4(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc >> 2);

  BEGIN_ASM()
VXORPD(YMM( 0), YMM( 0), YMM( 0))
VXORPD(YMM( 1), YMM( 1), YMM( 1))
VXORPD(YMM( 2), YMM( 2), YMM( 2))
VXORPD(YMM( 3), YMM( 3), YMM( 3))
VXORPD(YMM( 4), YMM( 4), YMM( 4))
VXORPD(YMM( 5), YMM( 5), YMM( 5))
VXORPD(YMM( 6), YMM( 6), YMM( 6))
VXORPD(YMM( 7), YMM( 7), YMM( 7))
VXORPD(YMM( 8), YMM( 8), YMM( 8))
VXORPD(YMM( 9), YMM( 9), YMM( 9))
VXORPD(YMM( 10), YMM( 10), YMM( 10))
VXORPD(YMM( 11), YMM( 11), YMM( 11))
VXORPD(YMM( 12), YMM( 12), YMM( 12))
VXORPD(YMM( 13), YMM( 13), YMM( 13))
VXORPD(YMM( 14), YMM( 14), YMM( 14))
VXORPD(YMM( 15), YMM( 15), YMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(YMM( 0), MEM(RAX, 0*8))
	VMOVUPD(YMM( 1), MEM(RAX, 4*8))

	VBROADCASTSD(YMM( 2), MEM(RBX, 0*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(YMM( 4), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 5), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 6), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 7), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 2*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(YMM( 8), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 9), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 10), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 11), YMM( 1), YMM( 3))
	VBROADCASTSD(YMM( 2), MEM(RBX, 4*8))
	VBROADCASTSD(YMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(YMM( 12), YMM( 0), YMM( 2))
	VFMADD231PD(YMM( 13), YMM( 1), YMM( 2))
	VFMADD231PD(YMM( 14), YMM( 0), YMM( 3))
	VFMADD231PD(YMM( 15), YMM( 1), YMM( 3))
	
	LEA(RAX, MEM(RAX, 8*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 5), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 7), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 9), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 11), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 13), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(YMM( 1), YMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), YMM( 1))
VADDPD(YMM( 1), YMM( 15), MEM(RCX, 4*8))
VMOVUPD(MEM(RCX, 4*8), YMM( 1))
	
	LEA(RCX, MEM(RCX, 8*8))


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
          "r13", "r14", "r15", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",  "memory"
  )



}


void dgemm_macro_kernel_avx2_16regs(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *  _A, double *  _B) {
    int64_t mp = mc / MR2;
    int64_t np = nc / NR2;
    int64_t nmcnc = 0;
    int64_t MR2NR2 = MR2*NR2;
    int64_t MR2KC = MR2*kc;
    int64_t NR2KC = NR2*kc;
    int64_t KC64 = (KC/64) * 64;
    int64_t KC32 = ((kc - KC64)/32) * 32;
    int64_t KC16 = ((kc - KC64 - KC32)/16) * 16;
    int64_t MR2KC64 = MR2 * KC64;
    int64_t NR2KC64 = NR2 * KC64;
    int64_t MR2KC32 = MR2 * KC32;
    int64_t NR2KC32 = NR2 * KC32;
    int64_t MR2KC16 = MR2 * KC16;
    int64_t NR2KC16 = NR2 * KC16;
    double *_A_p;
    double *_B_p;
    double *_C_p;
    int i,j,k;

#pragma omp parallel 
{
#pragma omp for private(i)
          for(j=0;j<np;++j) {
              for(i=0;i<mp;++i) {
                  #pragma forceinline
                  dgemm_kernel_avx2_16regs_asm_unroll2(kc  , &_A[i*MR2KC], &_B[j*NR2KC], &C[(j*mp + i)*MR2NR2], incRowC, incColC);
              }
          }
}
    return;

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
                    //_C_p = C + nmcnc*MR2NR2;
                    dgemm_kernel_avx2_16regs_asm_unroll0(KC16, &_A[i*MR2KC], &_B[j*NR2KC], &C[(j*mp +i)*MR2NR2], incRowC, incColC);
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
                      _C_p = C + nmcnc*MR2NR2;
                      dgemm_kernel_avx2_16regs_asm_unroll2(KC32, &_A[i*MR2KC], &_B[j*NR2KC], _C_p, incRowC, incColC);
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
                      _C_p = C + nmcnc*MR2NR2;

                      dgemm_kernel_avx2_16regs_asm_unroll2(KC32, &_A[i*MR2KC], &_B[j*NR2KC], _C_p, incRowC, incColC);

                      dgemm_kernel_avx2_16regs_asm_unroll0(KC16, &_A[i*MR2KC + KC32*MR2], &_B[j*NR2KC + KC32*NR2], _C_p, incRowC, incColC);

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
                  dgemm_kernel_avx2_16regs_asm_unroll4(KC64, &_A[i*MR2KC], &_B[j*NR2KC], &C[(j*mp + i)*MR2NR2], incRowC, incColC);
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
                  _C_p = C + nmcnc*MR2NR2;

                  dgemm_kernel_avx2_16regs_asm_unroll4(KC64, &_A[i*MR2KC], &_B[j*NR2KC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx2_16regs_asm_unroll2(KC32, &_A[i*MR2KC + MR2KC64], &_B[j*NR2KC + NR2KC64], _C_p, incRowC, incColC);

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
                  _C_p = C + nmcnc*MR2NR2;

                  dgemm_kernel_avx2_16regs_asm_unroll4(KC64, &_A[i*MR2KC], &_B[j*NR2KC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx2_16regs_asm_unroll0(KC16, &_A[i*MR2KC + KC64*MR2], &_B[j*NR2KC + KC64*NR2], _C_p, incRowC, incColC);

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
                  _C_p = C + nmcnc*MR2NR2;

                  dgemm_kernel_avx2_16regs_asm_unroll4(KC64, &_A[i*MR2KC], &_B[j*NR2KC], _C_p, incRowC, incColC);

                  dgemm_kernel_avx2_16regs_asm_unroll2(KC32, &_A[i*MR2KC + KC64*MR2], &_B[j*NR2KC + KC64*NR2], _C_p, incRowC, incColC);

                  dgemm_kernel_avx2_16regs_asm_unroll0(KC16, &_A[i*MR2KC + KC64*MR2 + KC32*MR2], &_B[j*NR2KC + KC64*NR2 + KC32*NR2], _C_p, incRowC, incColC);

                  nmcnc = nmcnc + 1;
              }
          }
}

        }
        break;
    }
}
