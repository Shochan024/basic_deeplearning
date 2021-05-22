#!-*-coding:utf-8-*-
import os
from tensorflow.keras.models import Sequential , model_from_json
from tensorflow.keras.layers import Conv2D , MaxPool2D
from tensorflow.keras.optimizers import Adam , SGD

from tensorflow.keras.layers import Dense , Activation , Dropout , Flatten , BatchNormalization , MaxPooling2D
from tensorflow.keras.utils import plot_model , to_categorical

from tensorflow.keras.datasets import cifar10

########################################
#           データを用意していく          #
########################################

( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical( y_train , 10 )
y_test = to_categorical( y_test , 10 )


########################################
#         ネットワークを構築していく       #
########################################


model = Sequential()
model.add( Conv2D( 64 , ( 3,3 ) , padding="same" , input_shape=(32,32,3) ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.3 ) )

model.add( Conv2D( 64 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( MaxPooling2D( pool_size=( 2,2 ) ) )

model.add( Conv2D( 64 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.3 ) )

model.add( Conv2D( 128 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 128 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( MaxPooling2D( pool_size=( 2,2 ) ) )

model.add( Conv2D( 256 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 256 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 256 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( MaxPooling2D( pool_size=( 2,2 ) ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( MaxPooling2D( pool_size=( 2,2 ) ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )
model.add( Dropout( 0.4 ) )

model.add( Conv2D( 512 , ( 3,3 ) , padding="same" ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( MaxPooling2D( pool_size=( 2,2 ) ) )
model.add( Dropout( 0.5 ) )

model.add( Flatten() )
model.add( Dense( 512 ) )
model.add( Activation("relu") )
model.add( BatchNormalization() )

model.add( Dropout( 0.5 ) )
model.add( Dense( 10 ) )
model.add( Activation("softmax") )

########################################
#               訓練してく               #
########################################

learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

model_yaml_file = "models/cnn_model.yaml"

if os.path.isfile( "models/cifar10vgg.h5" ):
    model.load_weights( "models/cifar10vgg.h5" )

sgd = SGD( lr=learning_rate , decay=lr_decay , momentum=0.9 , nesterov=True )

model.compile( loss="categorical_crossentropy" , optimizer=sgd , metrics=["accuracy"] )

#histroy = model.fit( X_train , y_train , epochs=10 )

open( model_yaml_file , "w" ).write( model.to_yaml() )
model.save_weights('models/cifar10vgg.h5')
