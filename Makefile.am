AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = -I m4 --install
AM_CFLAGS = $(CFLAGS) #-qmkl=sequential -O2 -g -march=core-avx2 -finline -finline-functions -inline-forceinline -restrict -ipo -qopenmp -lpthread

src_main_c = src/main.c src/qmckl_dgemm.c src/utils.c src/kernel_avx2_16regs.c 
src_main_f = src/qmckl_dgemm_f.f90
src_main_h = include/qmckl_dgemm.h include/qmckl_dgemm_private.h include/utils.h src/kernel_avx2_16regs.h src/bli_x86_asm_macros.h
test_main_c = tests/test_main.c
test_main_f = tests/test.f90

bin_PROGRAMS = qmckldgemm
qmckldgemm_SOURCES = $(src_main_c) #$(src_main_h)
include_HEADERS = $(src_main_h)
qmckldgemm_CFLAGS = $(AM_CFLAGS)
qmckldgemm_CFLAGS += "-I${top_srcdir}/include"  "-I${top_srcdir}/src"
qmckldgemm_LDFLAGS = -qmkl=sequential

lib_LTLIBRARIES = src/libqmckldgemm.la
src_libqmckldgemm_la_SOURCES = $(src_main_c) $(src_main_f)
src_libqmckldgemm_la_CPPFLAGS = "-I${top_srcdir}/src"
src_libqmckldgemm_la_CPPFLAGS += "-I${top_srcdir}/include"

check_PROGRAMS = tests/test_main tests/test_f
tests_test_main_SOURCES = $(test_main_c)
tests_test_main_LDADD   = src/libqmckldgemm.la
tests_test_main_LDFLAGS = -no-install
tests_test_main_CFLAGS = $(AM_CFLAGS)
tests_test_main_CPPFLAGS = "-I${top_srcdir}/src"
tests_test_main_CPPFLAGS += "-I${top_srcdir}/include"

tests_test_f_SOURCES = $(test_main_f)
tests_test_f_LDADD   = src/libqmckldgemm.la
tests_test_f_LDFLAGS = -no-install
tests_test_f_CFLAGS = $(AM_CFLAGS)
tests_test_f_CPPFLAGS = "-I${top_srcdir}/src"
tests_test_f_CPPFLAGS += "-I${top_srcdir}/include"

TESTS = tests/test_main tests/test_f

