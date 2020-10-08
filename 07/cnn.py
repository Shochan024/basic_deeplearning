import os
from tensorflow.keras.models import Sequential , model_from_json
from tensorflow.keras.layers import Conv2D , MaxPool2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense , Activation , Dropout , Flatten
from tensorflow.keras.utils import plot_model , to_categorical

from tensorflow.keras.datasets import cifar10

( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data()

nb_classes = 10

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical( y_train , nb_classes )
y_test = to_categorical( y_test , nb_classes )

model = Sequential()
model.add( Conv2D( 32 , 3 , input_shape=(32,32,3) ) )
model.add( Activation("relu") )
model.add( Conv2D(32,3) )
model.add( Activation("relu") )
model.add( MaxPool2D( pool_size=(2,2) ) )

model.add( Conv2D( 64,3 ) )
model.add( Activation("relu") )
model.add( MaxPool2D( pool_size=(2,2) ) )

model.add( Flatten() )
model.add( Dense( 1024 ) )
model.add( Activation("relu") )
model.add( Dropout(0.8) )

model.add( Dense( nb_classes , activation="softmax" ) )

adam = Adam(lr=1e-4)

model.compile( optimizer=adam , loss="categorical_crossentropy" ,\
 metrics=["accuracy"] )

history = model.fit( X_train , y_train , batch_size=10 , epochs=1 ,\
 verbose=1 , validation_split=0.1 )

plot_model( model , to_file="conv.png" )

print( model.summary() )

json_string = model.to_json()
open("cnn_model.json", 'w').write(json_string)

model.save_weights(os.path.join('./', 'cnn_model_weight.hdf5'))

json_string = open("cnn_model.json").read()
model = model_from_json(json_string)

model.load_weights( "cnn_model_weight.hdf5" )

y = model.predict( X_test )
print( y )

