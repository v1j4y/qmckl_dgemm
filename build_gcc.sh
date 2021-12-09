#!/bin/bash

#CC=${CC:-icpc}
CC=gcc
CCFLAGS=(
 -O3 
 -g
 -march=core-avx2
 -finline
 -finline-functions
 -m64  -I"${MKLROOT}/include"
)
# -DMKL_DIRECT_CALL_SEQ_JIT
# -march=native
# -xCORE-AVX2
# -qopt-zmm-usage=high
# -mcmodel=medium
# -DMKL_DIRECT_CALL_SEQ_JIT
#-I/home/vijayc/gemm_asm/MIPP/src -I.
MKL_PATH=/share/apps/intel/oneapi/mkl/2021.1.1/
OMP_PATH=/share/apps/intel/oneapi/mkl/2021.1.1/
MKL_PATH=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
MKL_LIB="-Wl,--start-group ${MKL_PATH}/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/lib/intel64/libmkl_intel_thread.a ${MKL_PATH}/lib/intel64/libmkl_core.a -Wl,--end-group"
MKL_LIB="-Wl,--start-group ${MKL_PATH}/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/lib/intel64/libmkl_intel_thread.a ${MKL_PATH}/lib/intel64/libmkl_core.a -Wl,--end-group"
MKL_LIB=" -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl"
#MKL_LIB="-mkl=sequential"
echo $MKL_PATH
echo $MKL_LIB

OBJ_CMD="-c "${CCFLAGS[@]}" utils.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

#OBJ_CMD="-c "${CCFLAGS[@]}" kernel.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp  "${LFLAGS[@]}" "${@}
#echo $CC $OBJ_CMD
#$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" kernel_sse2_8regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" kernel_avx2_8regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

#OBJ_CMD="-c "${CCFLAGS[@]}" kernel_sse2_16regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp  "${LFLAGS[@]}" "${@}
#echo $CC $OBJ_CMD
#$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" kernel_avx2_16regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" kernel_avx2_12x4_16regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

#OBJ_CMD="-c "${CCFLAGS[@]}" kernel_avx2_32regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp  "${LFLAGS[@]}" "${@}
#echo $CC $OBJ_CMD
#$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" main.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
BUILD_CMD="-o xtest "${CCFLAGS[@]}" main.c kernel_avx2_8regs.o kernel_sse2_8regs.o kernel_avx2_12x4_16regs.o kernel_avx2_16regs.o utils.o -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -fopenmp "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
echo $CC $BUILD_CMD
$CC $OBJ_CMD
$CC $BUILD_CMD
