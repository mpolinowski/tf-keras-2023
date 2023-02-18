import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.datasets import imdb

# loading an excerpt from the imdb dataset
top_words = 5000
(Xtrain, Ytrain), (Xtest, Ytest) = imdb.load_data(num_words=top_words)

# truncate movie reviews > 500 words
max_review_size = 500
Xtrain = sequence.pad_sequences(Xtrain, maxlen=max_review_size)
Xtest = sequence.pad_sequences(Xtest, maxlen=max_review_size)


# building the model
## number of feature outputs
feature_length = 32

model = Sequential()
model.add(Embedding(top_words, feature_length, input_length=max_review_size))
model.add(LSTM(100))
## output binary classification
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()


# training the model
model.fit(Xtrain, Ytrain, epochs=10, batch_size=128)

## training validation
val_loss, val_accuracy = model.evaluate(Xtest, Ytest)
print(val_loss, val_accuracy)