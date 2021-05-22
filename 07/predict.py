import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import plot_model , to_categorical
from tensorflow.keras.datasets import cifar10

model_yaml_file = "models/cnn_model.yaml"

labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

with open( model_yaml_file, 'r' ) as yaml_file:
    yaml_string = yaml_file.read()

model = model_from_yaml( yaml_string )
model.load_weights( "models/cifar10vgg.h5" )

#print( model.summary() )

( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0

axes = []
fig = plt.figure( tight_layout=True )
for i in range( 10 ):
    index = np.random.randint( 0 , X_test.shape[0] )
    correct = y_test[index]
    img = X_test[index]
    result = model.predict( img.reshape( 1 , 32 , 32 , 3 ) )
    axes.append( fig.add_subplot( 2,5,i+1 , xlabel="c:{} r:{}".format( labels[ correct[0] ] , labels[ result.argmax() ] )  ) )
    axes[i].imshow( img )
plt.savefig("models/results.png")
