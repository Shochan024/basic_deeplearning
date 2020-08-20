#!-*-coding:utf-8-*-
import numpy as np

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )

def error( y , t ):
    return 0.5 * np.sum( ( y - t )**2 )

classes = ["dog" , "cat"]
t = np.array([1,0])
X_0 = np.array([0.3,0.2,0.1]) #1を意味する特徴ベクトル
W_1 = np.random.randn(64,3).T
u_1 = np.dot( X_0 , W_1 )

X_1 = sigmoid( u_1 )
W_2 = np.random.randn( 2,64 ).T
u_2 = np.dot( X_1 , W_2 )

y = list( softmax( u_2 ) )
result = y.index( max( y ) )

e = error( y , t )
print( "Output layer :" , y )
print( "Classes :" , classes )
print( "Predict :" , classes[result] )
print( "Error :", e )
