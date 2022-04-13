# Installing

## Using MKL

```
./configure  --enable-mkl=yes --with-mkl-dir=$MKLROOT --enable-best-link=no CC=icc FC=ifort 
```

## Using OpenBLAS

```
./configure  --enable-openblas=yes --with-openblas-libdir=$CONDA_PREFIX --with-openblas-incdir=$CONDA_PREFIX --enable-best-link=no CC=gcc FC=gfortran
```
