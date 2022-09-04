from cProfile import label
from cgi import test
import datetime as dt
from statistics import mode
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

#Loading data
company_ticker = 'BEL.NS'

start = dt.datetime(2012,1,1)
end = dt.datetime(2022,1,1)

data = pdr.DataReader(company_ticker, 'yahoo', start, end)

#Preparing data
scaling  = MinMaxScaler(feature_range=(0,1))
scaled_data = scaling.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 50


xTraining = []
yTraining = []

for i in range(prediction_days, len(scaled_data)):
    xTraining.append(scaled_data[i-prediction_days:i,0])
    yTraining.append(scaled_data[i,0])

xTraining = np.array(xTraining)
xTraining = np.reshape(xTraining,(xTraining.shape[0], xTraining.shape[1], 1))
yTraining = np.array(yTraining)

#Building model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(xTraining.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xTraining,yTraining,epochs=100,batch_size=100)

#Testing
#Load test data

test_start = dt.date(2022,1,1)
test_end = dt.datetime.now()

test_data = pdr.DataReader(company_ticker, 'yahoo', test_start, test_end)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaling.transform(model_inputs)

#Test prediction

xTest = []

for x in  range(prediction_days,len(model_inputs)):
    xTest.append(model_inputs[x-prediction_days:x,0])

xTest = np.array(xTest)
xTest = np.reshape(xTest,(xTest.shape[0], xTest.shape[1], 1))

prediction_price = model.predict(xTest)
prediction_price = scaling.inverse_transform(prediction_price)

#Visual plot test prediction

plt.title(f"{company_ticker} predictions")
plt.plot(actual_prices, color="black", label="Actual price")
plt.plot(prediction_price, color="blue", label="Predicted price")
plt.xlabel('time')
plt.ylabel('share price')
plt.legend()
plt.show()

# Future prediction

real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0], real_data.shape[1], 1))

future_prediction = model.predict(real_data)
future_prediction = scaling.inverse_transform(future_prediction)
print(f"{company_ticker} future prediction: {future_prediction}")