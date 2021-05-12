#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt

def activate( u ):
    #シグモイド関数
    return 1 / ( 1 + np.exp( -u ) )

def activate_prime( u ):
    #活性化関数の導関数 シグモイド関数
    return activate( u ) * ( 1 - activate( u ) )

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )


def E( y , t ):
    #二乗和誤差
    return 1/2 * np.sum( ( y-t )**2 )

def E_prime( y , t ):
    return y - t


def forward( X1 ):
    global W2
    global W3
    z1 = np.dot( W2 , X1 )
    print( z1 )
    sys.exit()
    X2 = activate( u=z1 )
    z2 = np.dot( W3 , X2 )
    X3 = softmax( x=z2 )

    return X3 , z2

def backward( y , t , z , x ):
    global W2
    global W3
    dout = E_prime( y=y , t=t )
    grad_W3 = np.dot( z , dout.T )

    d_hidden = np.dot( W3.T , dout ) * activate_prime( u=z )
    grad_W2 = np.dot( x.T , d_hidden )

    W2 -= 0.0001 * grad_W2
    W3 -= 0.0001 * grad_W3

def learn( epochs=100 ):
    global T
    global X1
    global W2
    global W3
    earray = []
    for i in range( epochs ):

        z1 = np.dot( W2 , X1 )
        X2 = activate( u=z1 )
        z2 = np.dot( W3 , X2 )
        y = softmax( x=z2 )
        e = E( y=y , t=T )


        
        dout = E_prime( y=y , t=T )
        grad_W3 = np.dot( X2 , dout.T )

        d_hidden = np.dot( W3.T , dout ) * activate_prime( u=z1 )
        grad_W2 = np.dot( X1 , d_hidden.T )

        W2 -= 0.0001 * grad_W2.T
        W3 -= 0.0001 * grad_W3.T
        earray.append( e )

    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.grid()
    plt.plot( np.arange( 0 , epochs , 1 ) , earray )
    plt.savefig("check.png")


#One Hot
T = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0],
    [1,0]
]).T
X1 = np.array([
    [300,0],
    [-200,500],
    [-300,100],
    [800,-100],
    [100,-500]
]).T
W2 = np.random.randn( 5,2 )
W3 = np.random.randn( 2,5 )
learn( epochs=50000 )
