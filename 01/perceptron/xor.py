#!-*-coding:utf-8-*-
import numpy as np

def step( x , theta=-0.6 ):
    return np.where( x >= theta , 1 , 0 )


def XOR( x_1 , x_2 ):
    X_1 = np.array([x_1,x_2])
    W_1 = np.array([-0.5,-0.5])
    u_1 = np.dot( W_1 , X_1 )

    X_2 = step( x=u_1 )
    print( X_2 )
    W_2 = np.array([0,0.3,-0.3])
    u_2 = np.dot( W_2 , X_2 )

    return u_2

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            print( i,j )
            GATE( i , j )
            #print( "x_1={},x_2={} : y={}".format( i , j, step( x=GATE( i , j ) ) ) )
            #print( "x_1={},x_2={} : u={}".format( i , j, GATE( i , j ) ) )

test( GATE=XOR )
