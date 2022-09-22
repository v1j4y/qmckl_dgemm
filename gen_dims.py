#!/usr/bin/env python

#for i in range(1,10):
#  for j in range(1,10):
#    print((i*16))
#    print((j*14))
#    print("------ ",i*16*j*14)

for i in range(16*14,256*14+1):
  print(i, " -- ", (i//224)*224)
