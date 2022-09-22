#!/bin/bash

for i in {192..3456..192}
do
  echo "-------------------- Start" >> test_cache.out
  echo $(( $i )) >> test_cache.out
  if [ $(( $i % 14)) -ne 0 ]
  then
    if [ $(( $i )) -lt 144 ]
    then
      echo $(($i)) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMC=$(( $i )) -DKC=$(( $i )) -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
      echo $(( (($i/14) + 1)*14 )) >> test_cache.out
      time ./xtest >> test_cache.out
    else
      MC=$(( (($i/2)/16)*16 ))
      if [ $(($MC)) -lt 256 ]
      then
        #echo $(( (($i/7)/16)*16 )) >> test_cache.out
        echo $(( (($i/2)/16)*16 )) >> test_cache.out
        ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
        echo $(( (($i/14) + 1)*14 )) >> test_cache.out
        time ./xtest >> test_cache.out
      else
        MC=256
        echo $(( $MC )) >> test_cache.out
        ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
        echo $(( (($i/14) + 1)*14 )) >> test_cache.out
        time ./xtest >> test_cache.out
      fi
    fi
    echo "-------------------- Done" >> test_cache.out
  else
    if [ $(( $i )) -lt 144 ]
    then
      echo $(($i)) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMC=$(($i)) -DKC=$(($i)) -DNC=$(( $i )) -DMR=16 -DNR=14
      echo $(( ($i/14)*14 )) >> test_cache.out
      time ./xtest >> test_cache.out
    else
      MC=224
      echo $(( $MC )) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
      echo $(( ($i/14)*14 )) >> test_cache.out
      time ./xtest >> test_cache.out
      #MC=$(( (($i/2)/16)*16 ))
      #if [ $(($MC)) -lt 256 ]
      #then
      #  echo $(( (($i/2)/16)*16 )) >> test_cache.out
      #  #./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$(( (($i/7)/16)*16 )) -DKC=$(( (($i/7)/16)*16 )) -DNC=$(( $i )) -DMR=16 -DNR=14
      #  ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
      #  echo $(( ($i/14)*14 )) >> test_cache.out
      #  time ./xtest >> test_cache.out
      #else
      #  MC=256
      #  echo $(( $MC )) >> test_cache.out
      #  ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/14) + 1)*14 )) -DMR=16 -DNR=14
      #  echo $(( ($i/14)*14 )) >> test_cache.out
      #  time ./xtest >> test_cache.out
      #fi
    fi
    echo "-------------------- Done" >> test_cache.out
  fi
  rm ./xtest
  #./build_dgemm.sh -DMAT_DIM=$(( 14*$i )) -DMC=256 -DKC=256 -DNC=$(( 14*$i )) -DMR=16 -DNR=14
  #echo "-------------------- Start" >> test_cache.out
  #echo $(( 14*$i )) >> test_cache.out
  #time ./xtest >> test_cache.out
  #echo "-------------------- Done" >> test_cache.out
done
