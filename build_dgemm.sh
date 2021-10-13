#!/bin/bash

CC=${CC:-icpc}
CCFLAGS=(
 -O3 
 -g
 -xCORE-AVX512
 -DMKL_DIRECT_CALL_SEQ_JIT
 -qopt-zmm-usage=high
 -finline
 -finline-functions
 -inline-forceinline
 -ipo
)
# -restrict
# -mcmodel=medium
# -DMKL_DIRECT_CALL_SEQ_JIT
#-I/home/vijayc/gemm_asm/MIPP/src -I.
MKL_PATH=/share/apps/intel/oneapi/mkl/2021.1.1/
OMP_PATH=/share/apps/intel/oneapi/mkl/2021.1.1/
MKL_LIB="-Wl,--start-group ${MKL_PATH}/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/lib/intel64/libmkl_intel_thread.a ${MKL_PATH}/lib/intel64/libmkl_core.a -Wl,--end-group"
MKL_LIB="-mkl=parallel"
echo $MKL_PATH
echo $MKL_LIB

OBJ_CMD="-c "${CCFLAGS[@]}" utils.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread  "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" kernel.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread  "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" main.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread  "${LFLAGS[@]}" "${@}
BUILD_CMD="-o xtest "${CCFLAGS[@]}" main.c kernel.o utils.o -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp  -lpthread "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
echo $CC $BUILD_CMD
$CC $OBJ_CMD
$CC $BUILD_CMD
