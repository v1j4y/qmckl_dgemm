#!/bin/bash

#CC=${CC:-icpc}
CC=icpc
CCFLAGS=(
 -O3
 -g
 -march=core-avx2
 -finline
 -finline-functions
 -inline-forceinline
 -restrict
 -ipo
)
# -DMKL_DIRECT_CALL_SEQ_JIT
# -march=native
# -xCORE-AVX2
# -qopt-zmm-usage=high
# -mcmodel=medium
# -DMKL_DIRECT_CALL_SEQ_JIT
#-I/home/vijayc/gemm_asm/MIPP/src -I.
MKL_PATH=/opt/intel/oneapi/mkl/2021.4.0/
OMP_PATH=/opt/intel/oneapi/mkl/2021.4.0/
#MKL_LIB="-Wl,--start-group ${MKL_PATH}/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/lib/intel64/libmkl_intel_thread.a ${MKL_PATH}/lib/intel64/libmkl_core.a -Wl,--end-group"
#MKL_LIB="-mkl -static-intel -no-intel-extensions -qopenmp-link=static -static-libgcc -static-libstdc++ -Wl,--start-group ${MKL_PATH}/lib/intel64/libmkl_intel_lp64.a ${MKL_PATH}/lib/intel64/libmkl_intel_thread.a ${MKL_PATH}/lib/intel64/libmkl_core.a -Wl,--end-group"
MKL_LIB="-qmkl=sequential"

OBJ_CMD="-c "${CCFLAGS[@]}" src/utils.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" src/qmckl_dgemm.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

##OBJ_CMD="-c "${CCFLAGS[@]}" src/kernel_sse2_8regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
###echo $CC $OBJ_CMD
##$CC $OBJ_CMD

##OBJ_CMD="-c "${CCFLAGS[@]}" src/kernel_avx2_8regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
###echo $CC $OBJ_CMD
##$CC $OBJ_CMD

OBJ_CMD="-c "${CCFLAGS[@]}" src/kernel_avx2_16regs.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
$CC $OBJ_CMD

echo "-------------------------"
OBJ_CMD="-c "${CCFLAGS[@]}" src/main.c -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
echo $CC $OBJ_CMD
BUILD_CMD="-o xtest "${CCFLAGS[@]}" src/main.c qmckl_dgemm.o utils.o kernel_avx2_16regs.o -I"${MKL_PATH}" "${MKL_LIB}" -L"${OMP_PATH}" -qopenmp -lpthread "${LFLAGS[@]}" "${@}
echo $CC $BUILD_CMD
$CC $OBJ_CMD
$CC $BUILD_CMD
