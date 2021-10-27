#!/bin/bash

for i in {128..8192..64}
do
  echo "-------------------- Start" >> test_cache.out
  echo $(( $i )) >> test_cache.out
  ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMC=$(( $i )) -DKC=$(( $i )) -DNC=$(( $i )) -DMR=16 -DNR=14
  { time ./xtest >> test_cache.out; } 2>>test_cache.out
  rm ./xtest
  #./build_dgemm.sh -DMAT_DIM=$(( 14*$i )) -DMC=256 -DKC=256 -DNC=$(( 14*$i )) -DMR=16 -DNR=14
  #echo "-------------------- Start" >> test_cache.out
  #echo $(( 14*$i )) >> test_cache.out
  #time ./xtest >> test_cache.out
  #echo "-------------------- Done" >> test_cache.out
done
