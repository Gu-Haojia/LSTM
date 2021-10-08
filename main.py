import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

dataframe = pd.read_csv('./1.csv', usecols=[0], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalization
scale = MinMaxScaler(feature_range=(0, 1))
# #### dataset = scale.fit_transform(dataset)

train_size = int(len(dataset) * 0.7)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

trainlist = scale.fit_transform(trainlist)
testlist = scale.fit_transform(testlist)


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# Set look_back
look_back = 1
trainX, trainY = create_dataset(trainlist, look_back)
testX, testY = create_dataset(testlist, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.save(os.path.join("DATA", "Test" + ".h5"))

# make predictions
# #### model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# un-normalization
trainPredict = scale.inverse_transform(trainPredict)
trainY = scale.inverse_transform(trainY)
testPredict = scale.inverse_transform(testPredict)
testY = scale.inverse_transform(testY)

# plot
plt.figure(dpi=200)
plt.plot(trainY, label="train")
plt.plot(trainPredict[1:], label="predict")
plt.title("Train Data")
plt.legend()
plt.show()

plt.figure(dpi=200)
plt.plot(testY, label="test")
plt.plot(testPredict[1:], label="predict")
plt.title("Predict Data")
plt.legend()
plt.show()
