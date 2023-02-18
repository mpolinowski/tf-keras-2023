import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# loading the fashion mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# inspecting the dataset
# print(x_train.shape)
# print(x_test.shape)

# print(y_train[666])
# plt.imshow(x_train[666])
# plt.show()

# data pre-processing
## reshape training images
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

## normalize training images
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# vectorize labels for the 10 categories from 0-9
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# print(y_train.shape)


# building the model
model = Sequential()
## convolutional layer + pooling
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
## randomly drop 25% of neurons to prevent overfitting
model.add(Dropout(0.25))
## flatten before dense layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
# output layer assigns probability of 10 classes
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()


# fitting the model
checkpoint_filepath = '/checkpoint/model.weights.best.hdf5'

model_checkpoint_callback = ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[model_checkpoint_callback])

# validation run
val_loss, val_score = model.evaluate(x_test, y_test)
print(val_loss, val_score)


# run prediction
pred = model.predict(x_test)
## show prediction probabilities
pred_max = np.argmax(pred, axis=1)
print(pred[666])
print(pred_max[666])

## show corresponding image
## reshaping data 28x28
## to be able to show the image
x = x_test[666].reshape(28, 28)
plt.imshow(x)
plt.show()


# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)