import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# using the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# inspecting the dataset
# print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)
# plt.imshow(x_train[666])
# print(y_train[666])
# plt.show()

# data pre-processing

## normalize training images
## start by creating a single column 28x28=>784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# print(x_train[666])
x_train /=255
x_test /=255


# vectorize labels for the 10 categories from 0-9
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# print(y_train[666])
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


# building the model
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()


# fitting the model
model.fit(x_train, y_train, batch_size=128, epochs=10)

# validation run
val_loss, val_score = model.evaluate(x_test, y_test)
# print(val_loss, val_score)
# 0.07639817893505096 0.9789999723434448

# run prediction
pred = model.predict(x_test)

## show prediction probabilities
print(pred[666])
## print label with highest probability
pred_max = np.argmax(pred, axis=1)
print(pred_max[666])

## show corresponding image
## reshaping data 784 => 28x28
## to be able to show the image
x = x_test[666].reshape(28, 28)
plt.imshow(x)
plt.show()