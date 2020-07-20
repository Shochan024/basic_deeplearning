#!-*-coding: utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt

def activate( x ):
    #活性化関数 シグモイド関数
    return 1 / ( 1 + np.exp( -x ) )

def activate_prime( x ):
    #活性化関数の導関数 シグモイド関数
    return x * ( 1 - x )

def softmax(x):
    return np.exp( x ) / np.sum( np.exp( x ) )

def E( y , t ):
    return 1/2 * np.sum( ( y - t ) **2 )

def E_prime( y , t ):
    return y - t

def forward( x ):
    global W2
    global W3
    u1 = X1.dot( W2 )
    X2 = activate( x=u1 )
    u2 = X2.dot( W3 )
    X3 = softmax( x=u2 )

    return X3 , u2

def backward( y , t , u , x ):
    global W2
    global W3
    dout = E_prime( y=y , t=t )
    grad_W3 = np.dot( u.T , dout )
    d_hidden = np.dot( dout , W3.T ) * activate_prime( x=u )
    print( d_hidden.shape )
    sys.exit()
    grad_W2 = np.dot( x.T , d_hidden )

    W2 -= 0.001 * grad_W2
    W3 -= 0.001 * grad_W3


def learn( x , t , epochs=20000 ):
    earray = []
    for i in range(epochs):
        y , u = forward( x=x )
        e = E( y=y , t=t )
        backward( y=y , t=t , u=u , x=x )
        earray.append(e)

    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.grid()
    plt.plot( list( range( epochs ) ) , earray )
    plt.show()

#One Hot
T = np.array([
    [1,0],
])
X1 = np.array([
    [0.3,0.2],
])

W2 = np.array([
    [0.1,0.3],
    [0.2,0.4]
])
W3 = np.array([
    [0.1,0.3],
    [0.2,0.4]
])
learn( x=X1 , t=T )
