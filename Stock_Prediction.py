#!/usr/bin/env python
# coding: utf-8

# # Google Stock Price Prediction

# In[1]:


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import torch
import yfinance as yf


# ## Data Collection

# In[2]:


# Retrieving data from Yahoo Finance API

ticker = "GOOG"
df = yf.download(ticker, start='2000-01-01', end='2024-06-30')
df


# In[3]:


plt.figure(figsize=(16,8))
plt.title('Close Price Movement')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in $', fontsize=18)
plt.show


# In[4]:


# Filtering target values from dataset

data = df.filter(['Close'])
dataset = data.values
len(dataset)


# In[5]:


# 70% of db will be for training

training_data_size = math.ceil(len(dataset)*.70)
training_data_size


# ## Scaling and Preprocessing

# In[6]:


# Scaling

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[7]:


train_data = scaled_data[0:training_data_size, :]
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(X_train)
        print(y_train)
        


# In[8]:


X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# ## LSTM Neural Network

# In[9]:


model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[10]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[11]:


model.fit(X_train, y_train, batch_size=1, epochs=1)


# In[12]:


test_data = scaled_data[training_data_size - 60: ,:]
X_test = []
y_test = dataset[training_data_size:, :]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    


# In[13]:


X_test = np.array(X_test)


# In[14]:


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


# In[15]:


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# ## Model Results

# In[16]:


rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(f'RMSE: {rmse}')


# In[17]:


r2 = r2_score(y_test, predictions)
print(f'R2 Score: {r2}')


# In[18]:


train = data[:training_data_size]
valid = data[training_data_size:]
valid.loc[:, 'predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model LM')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'predictions']])
plt.legend(['Train','Val','predictions'], loc='lower right')
plt.show


# In[19]:


valid


# ## Prediction based on 5 day lookback

# In[20]:


# Fetching recent data through Yahoo Finance API

def fetch_recent_data(ticker, lookback_period):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_period + 10)

    recent_data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    return recent_data


ticker = "GOOG"
lookback_period = 5
recent_data = fetch_recent_data(ticker, lookback_period)
recent_data


# In[21]:


def predict_stock_price(model, recent_data, scaler, lookback_period=60):
    # Convert to numpy array
    if isinstance(recent_data, pd.DataFrame):
        recent_data = recent_data.values
    
    # Scale recent data
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    
    # Padding
    if len(recent_data_scaled) < lookback_period:
        padding = np.full((lookback_period - len(recent_data_scaled), 1), recent_data_scaled[0])
        recent_data_scaled = np.concatenate((padding, recent_data_scaled), axis=0)
    
    # Reshape recent data
    X = recent_data_scaled.reshape((1, lookback_period, 1))
    
    # Make prediction
    if hasattr(model, 'predict'):
        prediction_scaled = model.predict(X).flatten()
    else:
        raise TypeError("Model must have a 'predict' method.")
    
    # Transform  prediction to original scale
    predicted_price = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
    
    return predicted_price


# In[22]:


predicted_price = (predict_stock_price(model, recent_data, scaler))
print(f'Stock Price Prediction: ${predicted_price:.2f}')


# In[ ]:




