#!-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
def scatter( rho , n ):
    # 相関を持つ2つの乱数を生成し、それをxとyとする
    x = np.random.randn( n )
    y = x * rho + np.sqrt(1-rho**2) * np.random.randn( n )
    return x , y
def identity( x ):
    return x
def get_grad( X , B , T ):
    Xt_X = np.dot( X.T , X )
    Xt_X_B = np.dot( Xt_X , B ).T
    Xt_T = np.dot( X.T , T )
    grad_B = Xt_X_B - Xt_T
    return grad_B.T
def error( y , t ):
    return 0.5 * np.sum( ( y - t )**2 )
def approximation_line( a , b , x ):
    return a*x + b
n = 100
rho = 0.7
eta =  0.001
epoch = 1000
X , T = scatter( rho=rho , n=n ) #これを最急降下法で学習し、相関係数と一致するか確認
B = np.random.randint( -20 , 20 , ( 2 , 1 ) ) #回帰係数をランダムに生成
approximation_line_x = np.arange(-10,10,0.1)
# レコーディングの際一度実験したあとで追加
X = X.reshape( n , 1 )
"""
最後に追加
"""
fig2 = plt.figure()
ax3 = fig2.add_subplot(111,ylim=(np.min(X),np.max(X)))
ax3.scatter( X , T , alpha=0.3 , c='turquoise' , edgecolors='turquoise' )
ax3.grid(c='gainsboro', zorder=9)
intercept = np.ones_like( X )
X = np.append( X , intercept , axis=1 )
ims = []
Bvalue_trace = []
result = []
for i in range( epoch ):
    Bvalue_trace.append( B[0] )
    y_hats = approximation_line( a=B[0] , b=B[0] , x=approximation_line_x )
    ims.append( ax3.plot( approximation_line_x , y_hats , color='orange' ) )
    # 入力層
    u_1 = np.dot( X , B ) #各データに対してyが予測される
    # 出力層
    Y = identity( u_1 )
    grad_B = get_grad( X=X , B=B , T=T )
    # 更新
    B = B - ( eta * grad_B )
    e = error( y = Y , t = T )
    result.append( e )
epochs = np.arange(0,epoch,1)
"""
# 最初はこっちを実装
plt.grid()
plt.plot( epochs , Bvalue_trace )
plt.show()
"""
fig = plt.figure(tight_layout=True) #tight_layout=Trueで、グラフが重なってしまう問題を回避
ax1 = fig.add_subplot(211,title='coefficient transition') #2行1列の1番目
ax2 = fig.add_subplot(212,title='error function value transition')
ax1.plot( epochs , Bvalue_trace , color='green' )
ax1.axhline(rho,linestyle='dashed',c='black')
ax1.grid(c='gainsboro', zorder=9)
ax2.plot( epochs , result , color='green' )
ax2.grid(c='gainsboro', zorder=9)
fig.savefig( 'graphs/fitting_transition.png' )
plt.cla()
anim = ArtistAnimation( fig2 , ims , interval = 1 , blit = True )
anim.save('graphs/fitting.gif')
plt.close()
