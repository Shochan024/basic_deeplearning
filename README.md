## Deep Learning基礎解答コード
1. [パーセプトロン](#perceptron)
2. [活性化関数](#activation)
3. [ニューラルネットワーク](#neuralnet)
4. [学習アルゴリズム](#learningalg)
5. [具体的な例](#examples)
6. [誤差逆伝播法](#backpropergation)
7. [畳み込み層/Pooling層](#convolution_and_pooling)
8. [一般画像の分類](#general_image_class)
9. [転移学習/Fine Tuning](#fine_tuning)

<a id="perceptron"></a>
# 1. パーセプロトン
複雑で難しそうに思えるDeep Learningも、一つ一つ見ていくと意外とシンプルです。Deep Learningの入り口、ニューラルネットワークの元となったパーセプトロンについて学びます

<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/01/perpeptron.png" alt="パーセプトロン" width="400px">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/01/multi_perceptron.png" alt="多層パーセプトロン" width="400px">
</div>

<a id="activation"></a>
# 2. 活性化関数
ニューロンから次のニューロンに信号が伝達することを発火と言います。活性化関数は、この発火を数学的にモデル化した関数になります。機械学習には欠かせない活性化関数について、様々な切り口で学んでいきます
<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/02/activation.png" alt="活性化関数" width="400px">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/02/activation_prime.png" alt="活性化関数導関数" width="400px">
</div>
<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/02/tanh_prime.png" alt="tanh導関数導出過程" width="800px" style="object-fit:cover;">
</div>

<a id="neuralnet"></a>
# 3. ニューラルネットワーク
本講義の本題、ニューラルネットワークについて学びます。パーセプトロンとの違いや、その演算方法、実装方法を学びます。機械学習を学ぶ上では避けて通れない線形代数学の復習にも軽く触れておきます
<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/03/dot.png" alt="内積" width="400px">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/03/neuralnet.png" alt="ニューラルネットの伝播" width="400px">
</div>

<a id="learningalg"></a>
# 4. 学習アルゴリズム
機械学習における”学習”の実態部分について学びます。データとモデルによる予測結果のギャップを定量化し、その差を最小化する最適化問題を考えます。最適化問題を解く際に登場する微分法の確認もしながらゆっくりと学んでいきましょう
<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/04/anim.gif" alt="線形回帰のFitting" width="400px">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/04/dc.gif" alt="誤差関数" width="400px">
</div>
<div align="center">
  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/04/grad.png" alt="勾配ベクトル" width="400px" style="object-fit:cover;">
</div>

<a id="examples"></a>
# 5. 具体的な例
それでは一通り準備が整ったので、ニューラルネットワークを用いて実際に問題を解いてみます。画像認識で最も骨が折れると言っても過言ではない画像のくり抜きを含めたフォーマット解析などは割愛してしまいますが、ニューラルネットワークを用いてモデルをトレーニングしてみます

<a id="backpropergation"></a>
# 6. 誤差逆伝播法
これまでの微分の方法論では数値微分を用いてきましたが、数値微分は計算量が多く、効率がよくありません。それが原因で学習速度が非常に遅かったように思います。今回学ぶ誤差逆伝播法( 別名BackPropagation Model )は、微分対象の変数以外は一旦定数とみなすという偏微分法の性質を利用し、計算コストを大幅にカットする方法です

<a id="convolution_and_pooling"></a>
# 7. 畳み込み層/Pooling層
画像認識では、同じオブジェクトを意味する画像でも位置ずれやエッジの差異が大きなノイズとなり得ます。それが原因で、予測精度に悪影響を及ぼすことは珍しくありません。そう言ったノイズを緩和する方法が今回学ぶ内容となります

<a id="general_image_class"></a>
# 8. 一般画像の分類
今回は、文字ではなく一般的な画像の分類を学んでいきます。一般的な画像は文字データよりも特徴量が多く、Class数も多いので学習には相応の工夫が必要です。その工夫については次回に詳しく学ぶとして、今回は深いことは考えずこの問題を扱ってみましょう

<a id="fine_tuning"></a>
# 9. 転移学習/Fine Tuning
前回は、多Classで特徴量の多いテーマを扱いました。そう言った性質から、学習にかなりの時間を要すること、汎化性能の高いモデルがなかなかできにくいこと、似たような分類モデルの横展開のしにくさを感じた方は多いと思います。今回は、こう言った問題を緩和する方法を学びます

  <img src="http://web.sfc.keio.ac.jp/~t13073si/basic_deeplearning/09/teni.png" alt="転移学習" width="400px" style="object-fit:cover;">
