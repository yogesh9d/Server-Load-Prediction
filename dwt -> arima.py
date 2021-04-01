# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pywt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure

mydata = pd.read_csv("FinalData.csv")

lst = mydata.iloc[0].to_numpy()
lst = lst[1:]

mydata = mydata.T # transpose of the data frame
# print(mydata)

ind = mydata.index.values.tolist()
ind = ind[1:]

# print (ind)
# print (lst)


# %%

(cA, cD) = pywt.dwt(lst, 'db2', 'smooth') # using db2 wavelet function to decompose data 
A = pywt.idwt(cA, None, 'db2', 'smooth') # using inverse wavelet to reconstruct linear components
D = pywt.idwt(None, cD, 'db2', 'smooth') # using inverse wavelet to reconstruct non-linear components
# lst_rec = A + D # reconstruction of lst i.e data before decomposition


# %%
d = {'data': lst, 'linear components': A, 'non-linear components': D} # constructing map with non-decomposed and decomposed data
df = pd.DataFrame(d) # map -> data frame
df.index = ind 

print (df)


# figure(num=None, figsize=(100, 10), dpi=80, facecolor='w', edgecolor='k')
pl = df.plot(figsize=(100, 10), grid=True)
plt.xlabel('Date')
plt.ylabel('Requests Per Day')

mpl.style.use('seaborn')
pl.set_title('Decomposition of Data Into Linear and Non-linear Components'.format('seaborn'), color='C0')


pl.legend()

plt.show()


# %%
# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# load dataset
# split into train and test sets
# X -> series values
# X = A[:100]
X = A
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
accuracy = list()
rmse = list()
# rolling forecast
for t in range(len(test)):
	model = ARIMA(history, order=(0, 1, 12))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
	# print ('t = ' + str(t) + ' len = ', str(len(predictions)))
	rmse.append(sqrt(mean_squared_error(test[:len(predictions)], predictions)))
	print ('rmse = ' + str(rmse[len(rmse)-1]))
	# accuracy[t] = sqrt(mean_squared_error(yhat, obs))
# evaluate forecasts
pyplot.plot(rmse, color='red')
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes validation
pyplot.plot(test, color = 'blue')
pyplot.plot(predictions, color='green')
pyplot.plot(rmse, color='red')
pyplot.show()


