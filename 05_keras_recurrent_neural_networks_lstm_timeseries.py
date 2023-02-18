import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler


# loading the passenger into a dataframe
df = pd.read_csv('data/international-airline-passengers.csv', usecols=[1], skipfooter=2, engine='python')
# df.plot()
# plt.show()

# pre-processing
dataset = df.values
dataset = dataset.astype('float32')

## normalize data to 0->1 range
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

## data split 70/30
train_size = int(len(dataset)*.70)
test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print(train.shape, test.shape)


# creating the timeseries datasets
def create_dataset(dataset, look_back=1):
      dataX, dataY = [], []
      for i in range(len(dataset) - look_back - 1):
            # input data
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            # output data
            b = dataset[(i+look_back), 0]
            dataY.append(b)
      return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))


# building the model
model = Sequential()

model.add(LSTM(4, input_shape=(1,1)))
model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()


# fitting the model
model.fit(trainX, trainY, batch_size=1, epochs=1000)


# run prediction
trainPred = model.predict(trainX)
testPred = model.predict(testX)

# undo normalization
trainPred = scaler.inverse_transform(trainPred)
testPred = scaler.inverse_transform(testPred)

trainY = scaler.inverse_transform([trainY])
trainX = scaler.inverse_transform([testY])


# prediction plot
look_back = 1

trainPredPlot = np.empty_like(dataset)
trainPredPlot[ :, : ] = np.nan
trainPredPlot[ look_back : len(trainPred) + look_back, : ] = trainPred

testPredPlot = np.empty_like(dataset)
testPredPlot[ :, : ] = np.nan
testPredPlot[ len(trainPred) + (look_back*2) + 1 : len(dataset) - 1, : ] = testPred

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredPlot)
plt.plot(testPredPlot)
plt.show()

