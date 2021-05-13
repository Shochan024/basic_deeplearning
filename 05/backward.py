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
    return np.exp( x ) / np.sum( np.exp( x ) , axis=0 )


def E( y , t ):
    #二乗和誤差
    return 1/2 * np.sum( ( y-t )**2 )

def E_prime( y , t ):
    return y - t

def accuracy( y , t ):
    # 認識精度
    correct_y_indexs = y.argmax( axis=0 )
    return np.sum( correct_y_indexs == t.argmax( axis=0 ) ) / len( correct_y_indexs )


def learn( eta=0.001 , epochs=100 ):
    global T
    global X1
    global W2
    global W3
    earray = []
    acc_array = []
    for i in range( epochs ):

        # 順伝播
        z1 = np.dot( W2 , X1 )
        X2 = activate( u=z1 )
        z2 = np.dot( W3 , X2 )
        y = softmax( x=z2 )

        e = E( y=y , t=T )
        acc = accuracy( y=y , t=T )

        # 逆伝播
        dout = E_prime( y=y , t=T )
        grad_W3 = np.dot( X2 , dout.T )

        d_hidden = np.dot( W3.T , dout ) * activate_prime( u=z1 )
        grad_W2 = np.dot( X1 , d_hidden.T )

        W2 -= eta * grad_W2.T
        W3 -= eta * grad_W3.T

        earray.append( e )
        acc_array.append( acc )

    fig = plt.figure()
    ax1 = fig.add_subplot( 211 , xlabel="epochs" , ylabel="error" )
    ax2 = fig.add_subplot( 212 , xlabel="epochs" , ylabel="Accuracy" )
    ax1.grid(c='gainsboro', zorder=9)
    ax1.plot( np.arange( 0 , epochs , 1 ) , earray )

    ax2.grid(c='gainsboro', zorder=9)
    ax2.plot( np.arange( 0 , epochs , 1 ) , acc_array )
    fig.savefig("graphs/loss_and_accuracy.png")


#One Hot
T = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0],
    [1,0]
]).T
X1 = np.array([
    [3,0],
    [-2,5],
    [-3,1],
    [8,-1],
    [1,-5]
]).T
W2 = np.random.randn( 5,2 )
W3 = np.random.randn( 2,5 )
learn( epochs=2000 )
