AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = -I m4 --install
AM_CFLAGS = -qmkl=sequential -O3 -g -march=core-avx2 -finline -finline-functions -inline-forceinline -restrict -ipo -qopenmp -lpthread

src_main_c = src/main.c src/qmckl_dgemm.c src/utils.c src/kernel_avx2_16regs.c 
src_main_h = src/qmckl_dgemm.h src/qmckl_dgemm_private.h src/utils.h src/kernel_avx2_16regs.h src/bli_x86_asm_macros.h

bin_PROGRAMS = qmckldgemm
qmckldgemm_SOURCES = $(src_main_c) $(src_main_h)
include_HEADERS = $(src_main_h)
#qmckldgemm_CFLAGS = ${my_CFLAGS}
qmckldgemm_LDFLAGS = -qmkl=sequential

lib_LTLIBRARIES = src/libqmckldgemm.la
src_libqmckldgemm_la_SOURCES = $(src_main_c) 
src_libqmckldgemm_la_CPPFLAGS = "-I${top_srcdir}/share"
src_libqmckldgemm_la_CPPFLAGS += "-I${top_srcdir}/include"

