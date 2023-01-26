# Long Short-Term Memory Stock Predictions

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# enter your personal api key her
polygon_api_key = 'wMOYC48xRp4pxKupyAjbBos01hCDOXCb'

#formatting ticker
ticker = input("Enter ticker name: ")
formatted_ticker = ticker.upper()

# today's date 
date_time = datetime.datetime.now()
today = date_time.strftime("%Y-%m-%d")

# Make a request to the Polygon.io API to get historical stock data
historical_info = f'https://api.polygon.io/v2/aggs/ticker/{formatted_ticker}/range/1/hour/2023-01-01/{today}?adjusted=true&sort=asc&apiKey={polygon_api_key}'
response = requests.get(historical_info)
data = json.loads(response.text)
closing_prices = [x['c'] for x in data['results']]


# Split the closing prices data into train and test sets
train, test = train_test_split(closing_prices, test_size=0.2, shuffle=True)
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(closing_prices):
	print("TRAIN:", train_index, "TEST:", test_index)
	train, test = closing_prices[train_index[0]], closing_prices[test_index[0]]

# Create the LSTM network
model = Sequential()
# model.add(LSTM(4, input_shape=(1, train.shape[1]), return_sequences=True))
model.add(LSTM(4))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the data
train = np.array([train])
train = np.reshape(train, (train.shape[0], 1, 1))

test = np.array([test])
test = np.reshape(test, (test.shape[0], 1, 1))

train = np.reshape(train, (train.shape[0], 1, 1))
test = np.reshape(test, (test.shape[0], 1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices = np.array(closing_prices)
closing_prices = closing_prices.reshape(-1, 1)
closing_prices = scaler.fit_transform(closing_prices)

# training the model
model.fit(train, test, epochs=100, batch_size=25, verbose=2)

# make predictions
predictions = model.predict(train, batch_size=25)

# inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

print("+=============================+")
print(f"prediction: {predictions}")
print("+=============================+")
