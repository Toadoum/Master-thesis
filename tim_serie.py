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
ts.head(6)

ts.describe()

from datetime import datetime
con=ts['Draw.up.date']
ts['Draw.up.date']=pd.to_datetime(ts['Draw.up.date'])
ts.set_index('Draw.up.date', inplace=True)
#check datatype of index
ts.index

#time series plot
plt.figure(figsize=(10, 6))
ts.daily_count.plot()
plt.title('Draw up count per day')
plt.ylabel('count per day', fontsize=15)
plt.xlabel('Draw up date',fontsize=15)
plt.savefig('tsplot.png')
plt.show()

#distribution plot
#plt.rc('xtick', labelsize=15) 
#plt.rc('ytick', labelsize=15) 
plt.figure(figsize=(10, 6))
plt.title('Distribution of the data')
sns.distplot(ts.daily_count, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.ylabel('Frequency',fontsize=15)
plt.xlabel('daily_count',fontsize=15)
plt.legend(fontsize="x-large")
plt.savefig('distplot.png')

#Moving average
# weekly baseline
ts['baseline'] = ts.daily_count.shift(1)

# moving averages
ts['MVA2'] = ts.daily_count.rolling(2).mean().shift(1)
ts['MVA4'] = ts.daily_count.rolling(4).mean().shift(1)
ts['MVA6'] = ts.daily_count.rolling(6).mean().shift(1)
ts['MVA8'] = ts.daily_count.rolling(8).mean().shift(1)
ts.head(9)

ts50 = ts.daily_count[:50]

#%%
fig = plt.figure(figsize=(13,8))
n = 2
l = ['A', 'B', 'C', 'D']
for i in range(1, 5):
    ax = plt.subplot(2, 2, i)
    plt.subplots_adjust(wspace=0.3, hspace=1.3)
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 
    ax.set_title('(%s) Plot of weekly count and MVA%d' % (l[i-1], n), pad=20)
    ax.set_ylabel('Daily count',fontsize=15)
    ts50.plot(alpha=0.7, lw=2)
    ts50.rolling(n).mean().shift(1).plot(lw=3)
    ax.set_xlabel('draw up date',fontsize=15)
    plt.legend(fontsize="x-large")
    plt.legend(['count', 'MVA%d' % n])
    n = n +2
    plt.savefig('mvaplot.png')
   #%% 
ts.drop(ts.index[:8], inplace=True)
ts.head(8)

#Prediction performance functon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prediction_performance(model, test_labels, test_features):
    # Use the model's predict method on the test data
    predictions = model.predict(test_features)

    # Calculate the mean absolute errors
    mae = round(mean_absolute_error(test_labels, predictions), 1)
    
    # Calculate the root mean square error
    rms = round(np.sqrt(mean_squared_error(test_labels, predictions)), 1)

    # Calculate mean absolute percentage error (MAPE)
    errors = abs(predictions - test_labels)
    mape = None
    if 0 not in test_labels:
        mape = round(np.mean(100 * (errors / test_labels)), 1)    
    
    return (mae, rms, mape)

#Setting Features and labels
# Labels are the values we want to predict
labels = np.array(ts.daily_count)

# Remove the labels from the features
# axis 1 refers to the columns
features = ts
features = features.drop(['daily_count'], axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

print('Features shape:\t', features.shape)
print('Labels shape:\t', labels.shape)

#Spliting data into train and test set
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, \
    test_labels = train_test_split(features, labels, test_size = 0.2, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:\t', train_labels.shape)
print('Testing Features Shape:\t', test_features.shape)
print('Testing Labels Shape:\t', test_labels.shape)

#Baseline prediction performance
# The baseline prediction 
baseline_preds = test_features[:, feature_list.index('baseline')]
#baseline_preds = baseline_preds.astype(int)

# Baseline errors, and display average baseline error
errors = abs(baseline_preds - test_labels)
print('MAE:\t', round(np.mean(errors), 1))

# Calculate the root mean square error
rms = np.sqrt(mean_squared_error(test_labels, baseline_preds))
print('RMSE:\t', round(rms, 1))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print('MAPE:\t', round(np.mean(mape), 1), '%.')

#Moving average prediction performance
for mva in ['MVA2', 'MVA4', 'MVA6', 'MVA8']:
    print('----- {} -----'.format(mva))
    # The baseline prediction 
    baseline_preds = test_features[:, feature_list.index(mva)]
    #baseline_preds = baseline_preds.astype(int)

    # Baseline errors, and display average baseline error
    errors = abs(baseline_preds - test_labels)
    print('MAE:\t', round(np.mean(errors), 2))

    # Calculate the root mean square error
    rms = np.sqrt(mean_squared_error(test_labels, baseline_preds))
    print('RMSE:\t', round(rms, 2))

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    print('MAPE:\t', round(np.mean(mape), 2), '%.')
#%%
#AR Model
f=ts['MVA8']
f

from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
# split dataset
X = f.values

#train, test = X,X

train, test = X[0:380], X[380:449]
# train autoregression
model1 = AR(train)
model1_fit = model1.fit()
window = model1_fit.k_ar
coef = model1_fit.params
# walk forward over time steps in test
history = list(train[len(train)-window:])
#history = [history[i] for i in range(len(history))]
predict = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predict.append(yhat)
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))

errors = abs(predict - test)
print('MAE:\t', round(np.mean(errors), 1))
error = np.sqrt(mean_squared_error(test, predict))
print('RMSE: %.3f' % error)
mape = 100 * (errors / test)
print('MAPE:\t', round(np.mean(mape), 2), '%.')
# plot
#test_dataframe=pd.DataFrame(index=f.index)
#test_dataframe['date']=test

#predictions_dataframe=pd.DataFrame(index=f.index)
#predictions_dataframe['date']=predictions

plt.figure(figsize=(10, 6))
plt.title('Plot of AR model')
plt.plot(X,'o-',color='green',label='data')
plt.plot(range(0,380),X[0:380], color='red', label='Training data')
plt.plot(range(380,449) ,predict, color='blue', label='Predicted data')
plt.xlabel('Draw.up.date.index')
plt.ylabel('MVA8')
plt.legend(fontsize="x-large")
plt.savefig('ar1.png')

plt.figure(figsize=(10, 6))
plt.plot(test,label='test data')
plt.plot(predict, color='red', label='predicted data')
plt.title('Plot of AR model')
plt.ylabel('count per day',fontsize=15)
plt.xlabel('Draw up date.index',fontsize=15)
plt.legend(fontsize="x-large")
plt.savefig('ar2.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(errors,label='Residuals')
plt.title('Residuals plot of AR model')
plt.ylabel('Residuals',fontsize=15)
plt.xlabel('Prdicted values',fontsize=15)
plt.legend(fontsize="x-large")
#plt.savefig('ar2.png')
plt.show()

#ARIMA model
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
X = f.values
#size = int(len(X)*0.1 ) 
#train, test = X[0:395], X
train, test = X[0:380], X[380:449]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,0,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
#errors
errors1 = abs(predictions - test)
print('MAE:\t', round(np.mean(errors1), 1))
error = np.sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % error)
mape = 100 * (errors / test)
print('MAPE:\t', round(np.mean(mape), 2), '%.')
# plot
#test_dataframe=pd.DataFrame(index=f.index)
#test_dataframe['date']=test

#predictions_dataframe=pd.DataFrame(index=f.index)
#predictions_dataframe['date']=predictions
plt.figure(figsize=(10, 6))
plt.title('Plot of ARIMA model')
plt.plot(X,'o-',color='green',label='data')
plt.plot(range(0,380),X[0:380], color='red', label='Training data')
plt.plot(range(380,449) ,predictions, color='blue', label='Predicted data')
plt.xlabel('Draw.up.date.index',fontsize=15)
plt.ylabel('MVA8',fontsize=15)
plt.legend(fontsize="x-large")
plt.savefig('arima1.png')

plt.figure(figsize=(10, 6))

#plt.plot(test_dataframe)
#plt.plot(predictions_dataframe, color='red')

plt.plot(test, label='test data')
plt.plot(predictions, color='red', label='predicted data')

plt.title('Plot of ARIMA model')
plt.ylabel('count per day',fontsize=15)
plt.xlabel('Draw up date.index',fontsize=15)
plt.legend(fontsize="x-large")
plt.savefig('arima2.png')
plt.show()

#plt.figure(figsize=(10, 6))
#plt.plot(errors1,label='Residuals')
#plt.title('Residuals plot of ARIMA model')
#plt.ylabel('Residuals',fontsize=15)
#plt.xlabel('Prdicted values',fontsize=15)
#plt.legend(fontsize="x-large")
#plt.savefig('ar2.png')
plt.show()

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(X, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())






model.plot_diagnostics(figsize=(18,12))
plt.savefig('autoari.png')
plt.show()

