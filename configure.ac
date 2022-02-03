# Initialize
AC_INIT([qmckldgemm], [0.1])
AM_PROG_AR
LT_INIT
AC_CONFIG_MACRO_DIR([m4])

# Generate the Makefile
AM_INIT_AUTOMAKE([foreign -Wall -Werror])

# Include macros
sinclude(acx_mkl.m4)
sinclude(ac_pthread.m4)
sinclude(acx_openblas.m4)


# Provide special options for INTEL MKL
# We force the use of icc
AC_MSG_CHECKING([whether INTEL's MKL is enabled])
AC_ARG_ENABLE(mkl,
	[AS_HELP_STRING([--enable-mkl],
	[Use INTELs MKL for solvers and FFTs (default = no)])],
        enable_icc="yes"
        CC="icc"
	AC_MSG_RESULT([yes]),
	AC_MSG_RESULT([no]))

AC_LANG(C)

AC_PROG_FC(FC)

# Generate config
AC_CONFIG_HEADERS([config.h])

# Provide special options for the MKL library
AC_ARG_WITH(mkl-dir,
	[AS_HELP_STRING([--with-mkl-dir=<MKL path>],
	[Provide an alternative path to the MKL library])])

# Provide special options for OpenBLAS
AC_MSG_CHECKING([whether OpenBLAS is enabled])
AC_ARG_ENABLE(openblas,
	[AS_HELP_STRING([--enable-openblas],
	[Use the OpenBLAS library instead of ATLAS (default = no)])],
	AC_MSG_RESULT([yes]),
	AC_MSG_RESULT([no]))
AC_ARG_WITH(openblas-libdir,
	[AS_HELP_STRING([--with-openblas-libdir=<OpenBLAS library path>],
	[Provide an alternative path to the OpenBLAS library])])
AC_ARG_WITH(openblas-incdir,
	[AS_HELP_STRING([--with-openblas-incdir=<OpenBLAS header dir>],
	[Provide an alternative path to the OpenBLAS header directory])])


# Set flags for multithreading
#n_pthreads=4
#AC_ARG_ENABLE(threads,
#	[AS_HELP_STRING([--enable-threads@<:@=<max_number_of_threads>@:>@],
#	[Enable multhreading (on with up to 4 threads by default)])],
#    if test "$enableval" = "no"; then
#      use_pthreads="no"
#    else
#      use_pthreads="yes"
#      if test "$enableval" != "yes"; then
#        n_pthreads=$enableval
#      fi
#    fi,
#    use_pthreads=yes
#    )

# Enable linking options for making the executable as portable as possible.
AC_MSG_CHECKING([best linking option])
AC_ARG_ENABLE(best-link,
	[AS_HELP_STRING([--enable-best-link],
	[Choose the right combination of static and dynamic linking to make \
the executable as portable as possible (default = no)])],
	AC_MSG_RESULT([yes]),
	AC_MSG_RESULT([no]))

################ Actions to complete in case of multhreading ################
#AC_DEFINE_UNQUOTED(THREADS_NMAX, $n_pthreads,[Maximum number of POSIX threads])
#if test "$use_pthreads" = "yes"; then
#  AC_MSG_CHECKING([for multithreading])
#  AC_MSG_RESULT([maximum of $n_pthreads thread(s)])
#  AC_DEFINE(USE_THREADS, 1, [Triggers multhreading])
## CC, CFLAGS and LIBS are system and compiler-dependent
#  ACX_PTHREAD
#  CC="$PTHREAD_CC"
#  [AM_CFLAGS="$AM_CFLAGS $PTHREAD_CFLAGS -D_REENTRANT"]
#  LIBS="$PTHREAD_LIBS $LIBS"
#fi
#AM_CONDITIONAL(USE_THREADS, test $use_pthreads = "no")

############ handle the INTEL MKL library ###########
if test "$enable_mkl" = "yes"; then
  AC_MSG_CHECKING([Before ACX Call])
  convlibs=""
  ACX_MKL($with_mkl_dir,no,$enable_best_link,$convlibs)
  AC_MSG_CHECKING([for the INTEL MKL])
  if test "$MKL_WARN" == ""; then
    AC_MSG_RESULT([yes])
  else
    AC_MSG_RESULT([no])
    AC_MSG_WARN([$MKL_WARN])
  fi
  AM_CFLAGS="$AM_CFLAGS $MKL_CFLAGS "
  FCFLAGS=" $MKL_CFLAGS "
  AM_LDFLAGS="$AM_LDFLAGS $MKL_LDFLAGS "
  LIBS="$LIBS $MKL_LIBS"
else
  if test "x$enable_openblas" = "xyes"; then
######## Handle the OpenBLAS library (linear algebra: BLAS + LAPACKe) ########
    ACX_OPENBLAS($with_openblas_libdir, $with_openblas_incdir, $use_pthreads, no,
      [
        AM_CFLAGS="$AM_CFLAGS $OPENBLAS_CFLAGS "
        AM_LDFLAGS="$AM_LDFLAGS $OPENBLAS_LDFLAGS "
        LIBS="$OPENBLAS_LIBS $LIBS"
        if test "$OPENBLAS_WARN" != ""; then
          AC_MSG_WARN([$OPENBLAS_WARN])
        fi
      ],
      AC_MSG_ERROR([$OPENBLAS_ERROR Exiting.])
    )
  fi
fi
##############################################################################

AC_SUBST(CFLAGS)
AC_SUBST(AM_CFLAGS)
AC_SUBST(AM_CPPFLAGS)
AC_SUBST(AM_LDFLAGS)

# Generate Makefile from Makefile.in
AC_CONFIG_FILES([Makefile])

# Done
AC_OUTPUT

echo \
"-------------------------------------------------

${PACKAGE_NAME} Version ${PACKAGE_VERSION} 

Prefix ........:  ${prefix}
CC ............:  ${CC}
FC ............:  ${FC}
CPPFLAGS ......:  ${CPPFLAGS}
CFLAGS ........:  ${CFLAGS}
FCFLAGS........:  ${FCFLAGS}
FC ............:  ${FC}
FCLAGS ........:  ${FCFLAGS}
LDFLAGS .......:  ${LDFLAGS}
LIBS ..........:  ${LIBS}

Package features:
  Intel ICC    ...........:  ${enable_icc}
  OpenBlas     ...........:  ${enable_openblas}
  MKL enabled  ...........:  ${enable_mkl}


Now type 'make @<:@<target>@:>@'
where the optional <target> is:
  all           - build C library and Fortran binding
  check	        - run unit tests
  install       - install the library 
--------------------------------------------------"
