#!-*-coding:utf-8-*-
import numpy as np

y_hat = np.array([1,1])
t = np.array([0.5,0.5])

#############################
#     for文を用いるパターン    #
#############################
e = 0
for i in range( len( y_hat ) ):
    e += ( y_hat[i] - t[i] )**2

e = 0.5 * e
#############################
#        numpyの場合         #
#############################
e_numpy = 0.5 * np.sum( ( y_hat - t )**2 )
print( e_numpy )
