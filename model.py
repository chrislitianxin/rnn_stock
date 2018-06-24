from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from matplotlib import pyplot as plt
import sklearn.preprocessing as prep
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import time
import os


def load_data(dir):
    return pd.read_csv('Data/all_stocks_5yr.csv')

def preprocess(data):
    features = ['close','volume','diff']
    stock = data[data['Name']=='AAPL']
    stock['diff'] =  abs(stock.high - stock.low)
    stock = stock[features]

    norm = prep.MinMaxScaler()
    normStock = norm.fit_transform(stock)
    AAPL = np.array(AAPL)

    ts = 5
    X,y = [],[]
    for i in range(normStock.shape[0]-ts-1):
        X.append(normStock[i:i+ts])
        y.append(stock[i+ts,0])
    X = np.array(X)
    y = np.array(y)
    return X,y

def model(data):
    model = Sequential()

    model.add(LSTM(64,input_shape = (5,3),return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(16,return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('relu'))

    start = time.time()
    model.compile(loss='mse',optimizer='adam')
    print(time.time()-start)

    lstm = model.fit(X_train,y_train,epochs = 30,batch_size=100,validation_split=0.2,shuffle=False)
    return lstm
