from __future__ import print_function
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage

np.random.seed(123)

# linear stack of neural network layers
from keras.models import Sequential

# core layers of keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

#utility functions from keras
from keras.utils import np_utils


# MNIST is already kept in keras
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


from matplotlib import pyplot as plt
plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# Normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print(Y_train.shape)
print(Y_train[:10])

# split into 10 disctinct class labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)



##  Sequential Model Format
model = Sequential()

# input layer
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# fully connected dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compiling kernal adding ADAM loss function
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fitting the model
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)


#scoring

score = model.evaluate(X_test, Y_test, verbose=0)