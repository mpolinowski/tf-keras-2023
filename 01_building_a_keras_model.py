from keras.models import Model, Sequential
from keras.layers import Input, Dense

# building a model with 3 layers
input_layer = Input(shape=(3,))
dense_layer1 = Dense(4)(input_layer)
dense_layer2 = Dense(4)(dense_layer1)
output = Dense(4)(dense_layer2)

model = Model(inputs = input_layer, outputs=output)
model = Sequential()

model.add(Dense(4, name='dense_layer1', input_shape=(3,)))
model.add(Dense(4, name='dense_layer2'))
model.add(Dense(1, name='output'))

model.summary()


# compiling the model for training
model.compile(optimizer='adam', loss='mean_squared_error', metrics=('accuracy'))


## training the model
# model.fit(X, y, batch_size=32, epochs=12)


## evaluate model performance
# model.evaluate(Xval, yval)
# model.predict(Xtest)