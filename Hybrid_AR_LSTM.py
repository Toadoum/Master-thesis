#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:37:44 2019

@author: aimsadmin
"""
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error



from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd


import numpy as np                       
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
%matplotlib inline
plt.style.use('ggplot')

def AR1(data):
    
    # split dataset
    X = data
    
    #train, test = X,X
    
    #train, test = X[0:380], X[380:449]
    train=X
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = list(train[len(train)-window:])
    #history = [history[i] for i in range(len(history))]
    predict = list()
    for t in range(len(train)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = train[t]
        predict.append(yhat)
        history.append(obs)
        
        
        #print('predicted=%f, expected=%f' % (yhat, obs))
    return train, predict

def compute_residual(train,predict):
    residual=np.array(train)-np.array(predict)
    return residual


def preprocessing(training_set):
    
    training_set = training_set.reshape(-1,1)
    #feature scaling
    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(feature_range=(0,1))
    training_set_scaled= sc.fit_transform(training_set[0:380])
    test_set_scaled= sc.fit_transform(training_set[332:449])
    
    
    #Creating a data structure with 48 timesteps and 1 output
    #train
    X_train=[]
    Y_train=[]
    for i in range(48, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-48:i,0])
        Y_train.append(training_set_scaled[i,0])
    X_train, Y_train=np.array(X_train), np.array(Y_train)
    
    #test
    X_test=[]
    Y_test=[]
    for i in range(48, len(test_set_scaled)):
        X_test.append(test_set_scaled[i-48:i,0])
        Y_test.append(test_set_scaled[i,0])
        
    X_test,Y_test=np.array(X_test),np.array(Y_test)
    
    #Reshaping
    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    return X_train,Y_train,X_test,Y_test

#initialising the RNN

def model_LSTM(X_train,Y_train,dropout_prop=0.0,nb_units=10,batchsize=16,epoch=100,optimizer='adam'):
  regressor= Sequential()

  #adding the first LSTM layer and some dropout regularization
  regressor.add(LSTM(units=nb_units, return_sequences = True , input_shape=(X_train.shape[1],1)))
  regressor.add(Dropout(dropout_prop))

  #adding a third LSTM layer and some Dropout regularisation
  regressor.add(LSTM(units=nb_units, return_sequences = True ))
  regressor.add(Dropout(dropout_prop))

  #adding a fourth LSTM layer and some Dropout regularisation
  regressor.add(LSTM(units=nb_units ))
  regressor.add(Dropout(dropout_prop))

  #adding the output layer
  regressor.add(Dense(units=1))
  #Compiling the RNN

  regressor.compile(optimizer= optimizer, loss= 'mean_squared_error')


  #Fitting the RNN to the training se
  history=regressor.fit(X_train, Y_train,validation_split=0.20, epochs=epoch, batch_size=batchsize)

  return regressor

def evaluate_error(model,X_test,Y_test):
  predicted_birth=model.predict(X_test)
  #predicted_birth= sc.inverse_transform(predicted_birth)
  #Y_test= sc.inverse_transform(np.array(Y_test).reshape(-1,1))
  errors = abs(predicted_birth - Y_test)
  MAE=round(np.mean(errors), 2)
  rmse =np.sqrt(mean_squared_error(Y_test,predicted_birth))
#  mape = 100 * (errors / Y_test)
#  MAPE=round(np.mean(mape), 2)
  return rmse, MAE


def train_model(X_train,Y_train):


    model_regressor=model_LSTM(X_train,Y_train,dropout_prop=0,nb_units=10,batchsize=16,epoch=100,optimizer='Adam')
    #rmse=evaluate_error(model=model_regressor,X_test=X_test,Y_test=Y_test)
    


    return model_regressor

def main():

    ts=pd.read_csv("ts.csv", delimiter=',')
    from datetime import datetime
    con=ts['Draw.up.date']
    ts['Draw.up.date']=pd.to_datetime(ts['Draw.up.date'])
    ts.set_index('Draw.up.date', inplace=True)
    #check datatype of index
    ts.index
    
    # weekly baseline
    ts['baseline'] = ts.daily_count.shift(1)
    
    # moving averages
    ts['MVA2'] = ts.daily_count.rolling(2).mean().shift(1)
    ts['MVA4'] = ts.daily_count.rolling(4).mean().shift(1)
    ts['MVA6'] = ts.daily_count.rolling(6).mean().shift(1)
    ts['MVA8'] = ts.daily_count.rolling(8).mean().shift(1)
    ts.head(9)
    
    ts.drop(ts.index[:8], inplace=True)
    ts.head(8)
    
    #distribution plot
    
    #Define the training set
    training_set = ts['MVA8'].values
    
    train,predict_val=AR1(training_set)
    
    residual= compute_residual(train,predict_val)
    
    X_train,Y_train,X_test,Y_test = preprocessing(residual)

    model1 = train_model(X_train,Y_train)
    
    RMSE,MAE=evaluate_error(model1,X_test,Y_test)
    
    print('RMSE: ',RMSE,' MAE: ',MAE)

    predicted_birth_residual=model1.predict(X_test)
    
    predicted_birth= predicted_birth_residual+ predict_val[380:449]
    

   # predicted_birth=sc.inverse_transform(predicted_birth)
    
    
    plt.figure(figsize=(10, 5))
    plt.title('Plot of Hybrid model')
    plt.plot(training_set,'o-',color='green',label='data')
    plt.plot(range(0,380),training_set[0:380], color='red', label='Training data')
    plt.plot(range(380,449) ,predicted_birth[0], color='blue', label='Predicted data')
    plt.xlabel('Draw.up.date.index')
    plt.ylabel('MVA8')
    plt.legend()
    plt.savefig('hybplot1.png')
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.title('Plot of LSTM model')
    plt.plot(range(380,449),sc.Y_test, color='red', label='Test data')
    plt.plot(range(380,449) ,predicted_birth[0], color='blue', label='Predicted data')
    plt.xlabel('Draw.up.date.index')
    plt.ylabel('MVA8')
    plt.legend()
    plt.savefig('hybplot2.png')
    plt.show()

    return training_set, predict_val[380:449],predicted_birth_residual
    
    
t,p,pb= main()