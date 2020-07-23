#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def sigmoid_prime( x ):
    y = sigmoid( x )
    return y * ( 1 - y )

x = np.arange( -5 , 5 , 0.1 )
plt.grid()
plt.plot( x , sigmoid( x ) )
plt.plot( x , sigmoid_prime( x ) , marker="o" , markevery = 5 )
plt.savefig( "./graphs/sigmoid.png" )
