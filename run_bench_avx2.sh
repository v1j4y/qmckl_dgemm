#!/bin/bash

for i in {192..3456..192}
do
  echo "-------------------- Start" >> test_cache.out
  echo $(( $i )) >> test_cache.out
  if [ $(( $i % 6)) -ne 0 ]
  then
    if [ $(( $i )) -lt 64 ]
    then
      echo $(($i)) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMC=$(( $i )) -DKC=$(( $i )) -DNC=$(( (($i/6) + 1)*6 )) -DMR=8 -DNR=6
      echo $(( (($i/6) + 1)*6 )) >> test_cache.out
      time ./xtest >> test_cache.out
    else
      MC=$(( (($i/2)/8)*8 ))
      if [ $(($MC)) -lt 256 ]
      then
        #echo $(( (($i/7)/8)*8 )) >> test_cache.out
        echo $(( (($i/2)/8)*8 )) >> test_cache.out
        ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/6) + 1)*6 )) -DMR=8 -DNR=6
        echo $(( (($i/6) + 1)*6 )) >> test_cache.out
        time ./xtest >> test_cache.out
      else
        MC=256
        echo $(( $MC )) >> test_cache.out
        ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/6) + 1)*6 )) -DMR=8 -DNR=6
        echo $(( (($i/6) + 1)*6 )) >> test_cache.out
        time ./xtest >> test_cache.out
      fi
    fi
    echo "-------------------- Done" >> test_cache.out
  else
    if [ $(( $i )) -lt 64 ]
    then
      echo $(($i)) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMC=$(($i)) -DKC=$(($i)) -DNC=$(( $i )) -DMR=8 -DNR=6
      echo $(( ($i/6)*6 )) >> test_cache.out
      time ./xtest >> test_cache.out
    else
      MC=192
      echo $(( $MC )) >> test_cache.out
      ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/6) + 0)*6 )) -DMR=8 -DNR=6
      echo $(( ($i/6)*6 )) >> test_cache.out
      time ./xtest >> test_cache.out
      #MC=$(( (($i/2)/8)*8 ))
      #if [ $(($MC)) -lt 256 ]
      #then
      #  echo $(( (($i/2)/8)*8 )) >> test_cache.out
      #  #./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$(( (($i/7)/8)*8 )) -DKC=$(( (($i/7)/8)*8 )) -DNC=$(( $i )) -DMR=8 -DNR=6
      #  ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/6) + 1)*6 )) -DMR=8 -DNR=6
      #  echo $(( ($i/6)*6 )) >> test_cache.out
      #  time ./xtest >> test_cache.out
      #else
      #  MC=256
      #  echo $(( $MC )) >> test_cache.out
      #  ./build_dgemm.sh -DMAT_DIM=$(( $i )) -DMAT_DIM_M=$(( ($i/MC)*MC )) -DMAT_DIM_K=$(( ($i/MC)*MC )) -DMC=$MC -DKC=$MC -DNC=$(( (($i/6) + 1)*6 )) -DMR=8 -DNR=6
      #  echo $(( ($i/6)*6 )) >> test_cache.out
      #  time ./xtest >> test_cache.out
      #fi
    fi
    echo "-------------------- Done" >> test_cache.out
  fi
  rm ./xtest
  #./build_dgemm.sh -DMAT_DIM=$(( 6*$i )) -DMC=256 -DKC=256 -DNC=$(( 6*$i )) -DMR=8 -DNR=6
  #echo "-------------------- Start" >> test_cache.out
  #echo $(( 6*$i )) >> test_cache.out
  #time ./xtest >> test_cache.out
  #echo "-------------------- Done" >> test_cache.out
done
