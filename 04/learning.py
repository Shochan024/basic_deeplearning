#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as ani

def error( y_hat , t ):
    return 0.5*np.sum( ( y_hat - t ) **2  )

def learning( X , t , save=False , eta=0.005 , epoch=100 ):
    global w
    e_array = []
    figs = []
    fig , ax = plt.subplots()
    #plt.plot( a , c , "o" )
    for i in range( epoch ):
        y_hat = np.dot( X , w )
        #im = ax.plot( a , y_hat )
        #figs.append( im )
        E = error( y_hat , t )
        print( E )
        e_array.append(E)
        grad = np.dot( np.dot( X.T , X ) , w ) - np.dot( X.T , t )
        w -= eta*grad

    """
    animation = ani( fig , figs , interval=1000 )
    if save:
        animation.save("anim.gif" , writer="imagemagick")
    else:
        plt.show()
    """

    return e_array


################################
#           初期化              #
################################
r = 0.7
eta = 0.001
epoch = 100
a = np.random.randn(1000,1)
b = np.random.randn(1000,1)
c = r*a + np.sqrt( 1 - r**2 ) * b
X = np.concatenate([ a , c ],1)

################################
#      　  modelの定義           #
################################
w = np.random.randn(2,1)
t = a*r

e_array = learning( X=X , t=t , save=False)
plt.grid()
plt.plot( np.arange( 0 , len(e_array) , 1 ) , e_array )
plt.show()
