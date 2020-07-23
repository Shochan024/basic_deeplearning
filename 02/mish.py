#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def relu( x ):
    return np.where( x >= 0 , x , 0 )

def mish( x ):
    soft_plus = np.log( 1 + np.exp( x ) )
    return x * np.tanh( soft_plus )

def mish_prime( x ):
    soft_plus = np.log( 1 + np.exp( x ) )
    tanh = np.tanh( soft_plus )
    sigmoid = 1 / ( 1 + np.exp( -x ) )
    return x*sigmoid * ( 1 - tanh**2 ) + tanh

x = np.arange( -5 , 5 , 0.1 )
plt.grid()
plt.plot( x , mish( x ) )
plt.plot( x , mish_prime( x ) )
plt.savefig("./graphs/mish.png")
