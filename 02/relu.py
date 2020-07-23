#!-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def relu( x ):
    return np.where( x >= 0 , x , 0 )


x = np.arange( -5 , 5 , 0.1 )
plt.grid()
plt.plot( x , relu( x ) )
plt.savefig("./graphs/relu.png")
