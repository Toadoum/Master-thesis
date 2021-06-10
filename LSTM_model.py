"""
Created on Tue Nov  5 14:01:57 2019

@author: Sakayo_Toadoum
"""

import numpy as np                       
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
import pylab

%matplotlib inline
plt.style.use('ggplot')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

SMALL_SIZE = 15
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
plt.figure(figsize=(10, 6))
plt.title('Distribution of the data with MVA8 as variable')
sns.distplot(ts.MVA8, hist_kws={'alpha': 0.1}, kde = True, kde_kws={'alpha': 1})
plt.ylabel('Frequency', fontsize=15)
plt.xlabel('MVA8', fontsize=15)
plt.savefig('lstm1plot.png')

#Define the training set
training_set = ts['MVA8'].values

training_set = training_set.reshape(-1,1)
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set[0:380])
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





#Building the RNN
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the RNN
regressor= Sequential()

#adding the first LSTM layer and some dropout regularization
regressor.add(LSTM(units=10, return_sequences = True , input_shape=(X_train.shape[1],1)))
#regressor.add(Dropout(0.2))

#adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=10, return_sequences = True ))
#regressor.add(Dropout(0.2))

#adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=10 ))
#regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units=1))
#Compiling the RNN
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')


#Fitting the RNN to the training set
history=regressor.fit(X_train, Y_train,validation_split=0.20, epochs=100, batch_size=16)
history.history
print(history.history.keys())


#prediction
X_test=np.array(X_test)
X_test_shape=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_birth=regressor.predict(X_test_shape)
predicted_birth=sc.inverse_transform(predicted_birth)
Y_test=sc.inverse_transform(np.array(Y_test).reshape(-1,1))

#epochs plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Mean Squared Error', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('lstm2plot.png')
plt.show()

#vizualising result
errors = abs(predicted_birth - Y_test)
MAE=round(np.mean(errors), 2)
rmse =np.sqrt(mean_squared_error(Y_test,predicted_birth))
mape = 100 * (errors / Y_test)
MAPE=round(np.mean(mape), 2)
print('MAE on the data: %.4f' %MAE)
print('RMSE on the data: %.4f' %rmse)
print('MAPE on the data: %.4f' %MAPE)
plt.figure(figsize=(10, 6))
plt.title('Plot of LSTM model')
plt.plot(training_set,'o-',color='green',label='data')
plt.plot(range(0,380),training_set[0:380], color='red', label='Training data')
plt.plot(range(380,449) ,predicted_birth, color='blue', label='Predicted data')
plt.xlabel('Draw.up.date.index', fontsize=15)
plt.ylabel('MVA8', fontsize=15)
plt.legend()
plt.savefig('lstm3plot.png')
plt.show()


#plot of test and predict
plt.figure(figsize=(10, 5))
plt.title('Plot of LSTM model')
plt.plot(range(380,449),Y_test, color='red', label='Test data')
plt.plot(range(380,449) ,predicted_birth, color='blue', label='Predicted data')
plt.xlabel('Draw.up.date.index', fontsize=15)
plt.ylabel('MVA8', fontsize=15)
plt.legend()
plt.savefig('lstm4plot.png')
plt.show()
