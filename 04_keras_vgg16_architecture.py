import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# defining the input shape
# starting with a 224x224 colour image
input_shape = (224, 224, 3)

# building the model
model = Sequential()
## convolutional layers with 64 filters + pooling => 224 x 224 x 64
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## randomly drop 25% of neurons to prevent overfitting
model.add(Dropout(0.25))
## convolutional layers with 128 filters + pooling => 112 x 112 x 128
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## randomly drop 10% of neurons to prevent overfitting
model.add(Dropout(0.10))
## convolutional layers with 256 filters + pooling => 56 x 56 x 256
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## randomly drop 10% of neurons to prevent overfitting
model.add(Dropout(0.10))
## convolutional layers with 256 filters + pooling => 28 x 28 x 512
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## randomly drop 10% of neurons to prevent overfitting
model.add(Dropout(0.10))
## convolutional layers with 256 filters + pooling => 14 x 14 x 512
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## randomly drop 10% of neurons to prevent overfitting
model.add(Dropout(0.10))
## flatten before dense layer => 1 x 1 x 4096
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.25))
# output layer assigns probability of 1000 classes
model.add(Dense(1000, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# using the pre-trained vgg16 instead of a fresh version
vgg16 = VGG16()

# load image for prediction
img = load_img('data/sombrero.jpg', target_size=(224, 224))
img = img_to_array(img)
img = img.reshape(1,224,224,3)

# run prediction
yhat = vgg16.predict(img)
label = decode_predictions(yhat)
prediction = label[0][0]
print(prediction[1])