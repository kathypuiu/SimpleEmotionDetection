import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().magic('matplotlib inline')

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize 
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# "Face" dataset
# Imag shape (64,64,3)
# Training: 600 pictures
# Test: 150 pictures

def Model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    # CONV -> BN -> RELU 
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    # FLATTEN X (convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
     model = Model(inputs = X_input, outputs = X, name='Model')
    return model
    
    return model

#  Train ->`model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`  
# 4. Test ->`model.evaluate(x = ..., y = ...)`  

Model = Model(X_train.shape[1:])
Model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
Model.fit(X_train, Y_train, epochs=40, batch_size=50)
preds =Model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# - Use blocks of CONV->BATCHNORM->RELU such as:
# X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
# X = BatchNormalization(axis = 3, name = 'bn0')(X)
# X = Activation('relu')(X)
# until your height and width dimensions are quite low and your number of channels quite large (≈32).  
# You can then flatten the volume and use a fully-connected layer.
# - Use MAXPOOL after such blocks.  It will help you lower the dimension in height and width.
# - Change your optimizer.
# - If you get memory issues, lower your batch_size (12 )
# - Run more epochs until you see the train accuracy no longer improves. 

#your data
img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(Model.predict(x))
Model.summary()
plot_model(Model, to_file='Model.png')
SVG(model_to_dot(Model).create(prog='dot', format='svg'))

