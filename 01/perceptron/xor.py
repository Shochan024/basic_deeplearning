#!-*-coding:utf-8-*-
import numpy as np

def step( x , theta=0 ):
    return np.where( x >= theta , 1 , 0 )


def XOR( x_1 , x_2 ):
    X_1 = np.array([x_1,x_2])
    W_1 = np.array([[-0.2,-0.2],[0.1,0.1]])
    b_1 = np.array([0.2,-0.1])
    u_1 = np.dot( W_1 , X_1 ) + b_1

    X_2 = step( x=u_1 , theta=0 ) #左側がNAND 右側がORのベクトルが出力される
    W_2 = np.array([0.5,0.5])
    u_2 = np.dot( W_2 , X_2 )

    return u_2

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            print( "x_1={},x_2={} : y={}".format( i , j, step( x=GATE( i , j ) , theta=0.8 ) ) )

test( GATE=XOR )
