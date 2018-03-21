import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
#import preprocessing 

#Reading poloniex api, period is in seconds, start is timestamp for 1/1/2017, pair is USDT ETH.
pair = "USDT_ETH"
start = "1483228800"
period = "300"

print('On charge data')
#Since the API gives a json file we directly transform it to pandas dataframe.
df = pd.read_json("https://poloniex.com/public?command=returnChartData&currencyPair="+pair+"&start="+start+"&end=9999999999&period="+period)

print('data chargée')

#On fait des séries par jour
dataset=[]
df=df.drop('date',axis=1)
df=df.drop('volume',axis=1)
df=df.drop('quoteVolume',axis=1)
for i in range(int(len(df)/288)):
    dataset.append(df[i*288:288*(i+1)].values)
dataset=np.array(dataset)


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size)][0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0][0])
	return np.array(data_X), np.array(data_Y)

# FOR REPRODUCIBILITY
np.random.seed(7)


# PREPARATION OF TIME SERIES DATASE
prices=dataset
prices = np.reshape(prices, (len(prices),288,5)) # 1664
print(prices)
#scaler = MinMaxScaler(feature_range=(0, 1))
#OHLC_avg = scaler.fit_transform(prices)
OHLC_avg=prices
# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

print("On commence l'entrainement")
# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(288, 5), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))
#model.add(Activation('sigmoid'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=40, batch_size=2, verbose=1)

# model.compile(loss='binary_crossentropy', optimizer='adam') # Try SGD, adam, adagrad and compare!!!
# model.fit(trainX, binTrainY, epochs=5, batch_size=64, verbose=1)



# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))