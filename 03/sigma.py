#!-*-coding:utf-8-*-

X = [1,3,5]
W = [2,4,6]

u = 0
for i in range( len( X ) ):
    u += X[i] * W[i]

print( u )
