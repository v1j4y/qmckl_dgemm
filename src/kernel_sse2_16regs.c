#include "kernel_sse2_16regs.h"
#include "utils.h"
#include "qmckl_dgemm.h"

#include "bli_x86_asm_macros.h"


void dgemm_kernel_sse2_16regs_asm_unroll0(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc);

  BEGIN_ASM()
VXORPD(XMM( 0), XMM( 0), XMM( 0))
VXORPD(XMM( 1), XMM( 1), XMM( 1))
VXORPD(XMM( 2), XMM( 2), XMM( 2))
VXORPD(XMM( 3), XMM( 3), XMM( 3))
VXORPD(XMM( 4), XMM( 4), XMM( 4))
VXORPD(XMM( 5), XMM( 5), XMM( 5))
VXORPD(XMM( 6), XMM( 6), XMM( 6))
VXORPD(XMM( 7), XMM( 7), XMM( 7))
VXORPD(XMM( 8), XMM( 8), XMM( 8))
VXORPD(XMM( 9), XMM( 9), XMM( 9))
VXORPD(XMM( 10), XMM( 10), XMM( 10))
VXORPD(XMM( 11), XMM( 11), XMM( 11))
VXORPD(XMM( 12), XMM( 12), XMM( 12))
VXORPD(XMM( 13), XMM( 13), XMM( 13))
VXORPD(XMM( 14), XMM( 14), XMM( 14))
VXORPD(XMM( 15), XMM( 15), XMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))

	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 5), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 7), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 9), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 11), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 13), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 15), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))


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
          "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",  "memory"
  )



}

void dgemm_kernel_sse2_16regs_asm_unroll2(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc >> 1);


  BEGIN_ASM()
VXORPD(XMM( 0), XMM( 0), XMM( 0))
VXORPD(XMM( 1), XMM( 1), XMM( 1))
VXORPD(XMM( 2), XMM( 2), XMM( 2))
VXORPD(XMM( 3), XMM( 3), XMM( 3))
VXORPD(XMM( 4), XMM( 4), XMM( 4))
VXORPD(XMM( 5), XMM( 5), XMM( 5))
VXORPD(XMM( 6), XMM( 6), XMM( 6))
VXORPD(XMM( 7), XMM( 7), XMM( 7))
VXORPD(XMM( 8), XMM( 8), XMM( 8))
VXORPD(XMM( 9), XMM( 9), XMM( 9))
VXORPD(XMM( 10), XMM( 10), XMM( 10))
VXORPD(XMM( 11), XMM( 11), XMM( 11))
VXORPD(XMM( 12), XMM( 12), XMM( 12))
VXORPD(XMM( 13), XMM( 13), XMM( 13))
VXORPD(XMM( 14), XMM( 14), XMM( 14))
VXORPD(XMM( 15), XMM( 15), XMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 5), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 7), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 9), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 11), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 13), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 15), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))


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
          "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",  "memory"
  )



}

void dgemm_kernel_sse2_16regs_asm_unroll4(int64_t kc, double *A, double *B, double *C, int64_t incRowC, int64_t incColC) {
    uint64_t kl = (kc >> 2);


  BEGIN_ASM()
VXORPD(XMM( 0), XMM( 0), XMM( 0))
VXORPD(XMM( 1), XMM( 1), XMM( 1))
VXORPD(XMM( 2), XMM( 2), XMM( 2))
VXORPD(XMM( 3), XMM( 3), XMM( 3))
VXORPD(XMM( 4), XMM( 4), XMM( 4))
VXORPD(XMM( 5), XMM( 5), XMM( 5))
VXORPD(XMM( 6), XMM( 6), XMM( 6))
VXORPD(XMM( 7), XMM( 7), XMM( 7))
VXORPD(XMM( 8), XMM( 8), XMM( 8))
VXORPD(XMM( 9), XMM( 9), XMM( 9))
VXORPD(XMM( 10), XMM( 10), XMM( 10))
VXORPD(XMM( 11), XMM( 11), XMM( 11))
VXORPD(XMM( 12), XMM( 12), XMM( 12))
VXORPD(XMM( 13), XMM( 13), XMM( 13))
VXORPD(XMM( 14), XMM( 14), XMM( 14))
VXORPD(XMM( 15), XMM( 15), XMM( 15))
MOV(RSI, VAR(k))
MOV(RAX, VAR(a))
MOV(RBX, VAR(b))
MOV(RCX, VAR(c))

    TEST(RSI, RSI)
    JE(K_LOOP)
	LABEL(LOOP1)


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


	VMOVUPD(XMM( 0), MEM(RAX, 0*8))
	VMOVUPD(XMM( 1), MEM(RAX, 2*8))

	MOVLPD(XMM( 2), MEM(RBX, 0*8))
	MOVHPD(XMM( 2), MEM(RBX, 0*8))
	MOVLPD(XMM( 3), MEM(RBX, 1*8))
	MOVHPD(XMM( 3), MEM(RBX, 1*8))
	VFMADD231PD(XMM( 4), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 5), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 6), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 7), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 2*8))
	MOVHPD(XMM( 2), MEM(RBX, 2*8))
	MOVLPD(XMM( 3), MEM(RBX, 3*8))
	MOVHPD(XMM( 3), MEM(RBX, 3*8))
	VFMADD231PD(XMM( 8), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 9), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 10), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 11), XMM( 1), XMM( 3))
	MOVLPD(XMM( 2), MEM(RBX, 4*8))
	MOVHPD(XMM( 2), MEM(RBX, 4*8))
	MOVLPD(XMM( 3), MEM(RBX, 5*8))
	MOVHPD(XMM( 3), MEM(RBX, 5*8))
	VFMADD231PD(XMM( 12), XMM( 0), XMM( 2))
	VFMADD231PD(XMM( 13), XMM( 1), XMM( 2))
	VFMADD231PD(XMM( 14), XMM( 0), XMM( 3))
	VFMADD231PD(XMM( 15), XMM( 1), XMM( 3))
	
	LEA(RAX, MEM(RAX, 4*8))
	LEA(RBX, MEM(RBX, 6*8))


    DEC(RSI)
    JNE(LOOP1)
    
LABEL(K_LOOP)

PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 4), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 5), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 6), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 7), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 8), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 9), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 10), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 11), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 12), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 13), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))
PREFETCH(0, MEM(RCX, 192*8))
VADDPD(XMM( 1), XMM( 14), MEM(RCX, 0*8))
VMOVUPD(MEM(RCX, 0*8), XMM( 1))
VADDPD(XMM( 1), XMM( 15), MEM(RCX, 2*8))
VMOVUPD(MEM(RCX, 2*8), XMM( 1))
	
	LEA(RCX, MEM(RCX, 4*8))


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
          "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",  "memory"
  )



}


void dgemm_macro_kernel_sse2_16regs(int64_t mc, int64_t kc, int64_t nc, double *C, int64_t incRowC, int64_t incColC, double *  _A, double *  _B) {
    int64_t mp = MC / MR1;
    int64_t np = nc / NR1;
    int64_t nmcnc = 0;
    int64_t MR1NR1 = MR1*NR1;
    int64_t MR1KC = MR1*KC;
    int64_t NR1KC = NR1*KC;
    int64_t KC64 = (KC/64) * 64;
    int64_t KC32 = ((kc - KC64)/32) * 32;
    int64_t KC16 = ((kc - KC64 - KC32)/16) * 16;
    int64_t MR1KC64 = MR1 * KC64;
    int64_t NR1KC64 = NR1 * KC64;
    int64_t MR1KC32 = MR1 * KC32;
    int64_t NR1KC32 = NR1 * KC32;
    int64_t MR1KC16 = MR1 * KC16;
    int64_t NR1KC16 = NR1 * KC16;
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
                    //_C_p = C + nmcnc*MR1NR1;
                    dgemm_kernel_sse2_16regs_asm_unroll0(KC16, &_A[i*MR1KC], &_B[j*NR1KC], &C[(j*mp +i)*MR1NR1], incRowC, incColC);
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
                      _C_p = C + nmcnc*MR1NR1;
                      dgemm_kernel_sse2_16regs_asm_unroll2(KC32, &_A[i*MR1KC], &_B[j*NR1KC], _C_p, incRowC, incColC);
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
                      _C_p = C + nmcnc*MR1NR1;

                      dgemm_kernel_sse2_16regs_asm_unroll2(KC32, &_A[i*MR1KC], &_B[j*NR1KC], _C_p, incRowC, incColC);

                      dgemm_kernel_sse2_16regs_asm_unroll0(KC16, &_A[i*MR1KC + KC32*MR1], &_B[j*NR1KC + KC32*NR1], _C_p, incRowC, incColC);

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
                  dgemm_kernel_sse2_16regs_asm_unroll4(KC64, &_A[i*MR1KC], &_B[j*NR1KC], &C[(j*mp + i)*MR1NR1], incRowC, incColC);
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
                  _C_p = C + nmcnc*MR1NR1;

                  dgemm_kernel_sse2_16regs_asm_unroll4(KC64, &_A[i*MR1KC], &_B[j*NR1KC], _C_p, incRowC, incColC);

                  dgemm_kernel_sse2_16regs_asm_unroll2(KC32, &_A[i*MR1KC + MR1KC64], &_B[j*NR1KC + NR1KC64], _C_p, incRowC, incColC);

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
                  _C_p = C + nmcnc*MR1NR1;

                  dgemm_kernel_sse2_16regs_asm_unroll4(KC64, &_A[i*MR1KC], &_B[j*NR1KC], _C_p, incRowC, incColC);

                  dgemm_kernel_sse2_16regs_asm_unroll0(KC16, &_A[i*MR1KC + KC64*MR1], &_B[j*NR1KC + KC64*NR1], _C_p, incRowC, incColC);

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
                  _C_p = C + nmcnc*MR1NR1;

                  dgemm_kernel_sse2_16regs_asm_unroll4(KC64, &_A[i*MR1KC], &_B[j*NR1KC], _C_p, incRowC, incColC);

                  dgemm_kernel_sse2_16regs_asm_unroll2(KC32, &_A[i*MR1KC + KC64*MR1], &_B[j*NR1KC + KC64*NR1], _C_p, incRowC, incColC);

                  dgemm_kernel_sse2_16regs_asm_unroll0(KC16, &_A[i*MR1KC + KC64*MR1 + KC32*MR1], &_B[j*NR1KC + KC64*NR1 + KC32*NR1], _C_p, incRowC, incColC);

                  nmcnc = nmcnc + 1;
              }
          }
}

        }
        break;
    }
}
