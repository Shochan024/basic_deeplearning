#!-*-coding:utf-8-*-
import numpy as np

def step( x , theta=-0.6 ):
    return np.where( x >= theta , 1 , 0 )


def NAND( x_1 , x_2 ):
    X = np.array([x_1,x_2])
    W = np.array([-0.5,-0.5])
    u = np.dot( X , W ) #内積をとる
    return u

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            print( "x_1={},x_2={} : y={}".format( i , j, step( x=GATE( i , j ) ) ) )
            #print( "x_1={},x_2={} : u={}".format( i , j, GATE( i , j ) ) )

test( GATE=NAND )
