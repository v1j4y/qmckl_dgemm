# Initialize
AC_INIT([qmckldgemm], [0.1])
AM_PROG_AR
LT_INIT
AC_CONFIG_MACRO_DIR([m4])

# Generate the Makefile
AM_INIT_AUTOMAKE([foreign -Wall -Werror])

AC_LANG(C)

# Make sure gcc is working
AC_PROG_CC([icpc])

# Make sure Fortran is available
AC_PROG_FC([ifort])

# Generate config
AC_CONFIG_HEADERS([config.h])

# Generate Makefile from Makefile.in
AC_CONFIG_FILES([Makefile])

# Flags
CFLAGS="-qmkl=sequential -O2 -g -march=core-avx2 -finline -finline-functions -inline-forceinline -restrict -ipo -qopenmp -lpthread"

# Done
AC_OUTPUT

echo \
"-------------------------------------------------

${PACKAGE_NAME} Version ${PACKAGE_VERSION} 

Prefix ........:  ${prefix}
CC ............:  ${CC}
CPPFLAGS ......:  ${CPPFLAGS}
CFLAGS ........:  ${CFLAGS}
FC ............:  ${FC}
FCLAGS ........:  ${FCFLAGS}
LDFLAGS .......:  ${LDFLAGS}
LIBS ..........:  ${LIBS}

Now type 'make @<:@<target>@:>@'
where the optional <target> is:
  all           - build C library and Fortran binding
  check	        - run unit tests
  install       - install the library 
--------------------------------------------------"
