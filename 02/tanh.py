#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def tanh( x ):
    sinh = ( np.exp( x ) - np.exp( -x ) ) / 2
    cosh = ( np.exp( x ) + np.exp( -x ) ) / 2

    return sinh / cosh

def tanh_prime( x ):
    return 1 - tanh( x ) ** 2

x = np.arange( -5 , 5 , 0.1 )
plt.grid()
plt.plot( x , tanh( x ) )
plt.plot( x , tanh_prime( x ) )
plt.savefig( "./graphs/tanh.png" )
