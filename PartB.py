# Abhay Singh Raihal
# B20144
# 9319337883

# importing required libraries
from cProfile import label
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR

# reading the csv file
series = pd.read_csv("daily_covid_cases.csv", parse_dates=['Date'], index_col=['Date'], sep=',')

test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
# splitting the data into train and test data
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
# defining the minimum value that autorcorrelation can take
minval = 2 / math.sqrt(len(train))

train1 = []
for i in range(len(train)):
    train1.append(train[i,0])

# finding the optimal heuristics lag value
i = 0
cor = 1
while (abs(cor) > minval):
    i+=1
    lag = np.array(train1[i:])
    orig = np.array(train1[:-i])
    cor = pearsonr(lag, orig)[0]

i-=1
print("Optimal heuristics lag value :", i)

# now we will use this optimal heusristic lag value to predict future covid wave
window = i # The lag=5
train = X
test = [i for i in range(90)]
model = AR(train, lags = window, old_names = False)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
# print("Coefficients obtained from AR model are : ", np.round(coef, 3).astype(str))

#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(90):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

plt.plot([i for i in range(len(X))], X, color = "red", label = "actual")  # plotting the scatterplot between actual and predicted test data
plt.plot([(i+len(X)) for i in range(len(test))], predictions, color = "green", label = "predicted")
plt.legend()
plt.show()

# # printing the root mean square percentage error and mean absolute percentage error
# rmspe_test = math.sqrt(mean_squared_error(test, predictions))/np.mean(test)*100
# print("RMSE % between actual and predicted test data : {} %".format(round(rmspe_test, 3)))
# mape_test = mean_absolute_percentage_error(test, predictions)*100
# print("MAPE % between actual and predicted test data : {} %".format(round(mape_test, 3)))



