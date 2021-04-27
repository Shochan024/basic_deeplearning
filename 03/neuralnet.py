#!-*-coding:utf-8-*-
import numpy as np

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )

def error( y , t ):
    return 0.5 * np.sum( ( y - t )**2 )


for i in range( 0 , 10 ):
    classes = ["dog" , "cat"]
    t = np.array([1,0]) #dogを意味する特徴ベクトルだとする one hotというスタイル
    X_0 = np.array([0.3,0.2,0.1]) #dogを意味する特徴ベクトル
    W_1 = np.random.randn(64,3).T #変わっているポイントはここだけなので、weightをうまく調節することでこれをdogだと正確に予測できるような方法を学んでいく
    u_1 = np.dot( X_0 , W_1 ) #ユニット

    X_1 = sigmoid( u_1 ) #活性化して、次の層に伝播させていく
    W_2 = np.random.randn( 2,64 ).T #変わっているポイントはここだけなので、weightをうまく調節することでこれをdogだと正確に予測できるような方法を学んでいく
    u_2 = np.dot( X_1 , W_2 )

    y = list( softmax( u_2 ) )
    result = y.index( max( y ) )

    e = error( y , t )
    print( "======{}回目の試行結果=======".format(i) )
    print( "Output layer :" , y )
    print( "Classes :" , classes )
    print( "Predict :" , classes[result] , "Correct Answer :", "dog" )
    print( "Status :" , classes[result] == "dog" )
    print( "Error :", e )
    print( "\n" )

#今回は特徴ベクトルを一つ分だけ
